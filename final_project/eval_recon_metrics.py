"""Conventional reconstruction metrics: PSNR / SSIM / LPIPS / rFID, plus NLL.

The paper reports rFID (as "ceiling") and LPIPS but not PSNR or SSIM, which are
expected in any autoencoder table, and no likelihood at all, which is the
conventional metric for the VAE rungs. This fills both gaps on held-out data.

On bits/dim: it is only well defined against a discrete likelihood over the 256
pixel values. Our decoders emit continuous tanh outputs, so we report the
discretised-Gaussian bound --- the calibrated observation variance integrated
over each pixel bin --- and label it as such rather than passing it off as an
exact bits/dim. For the NVAE the KL term is available, so its number is a real
(bound on the) negative log-likelihood; for the autoencoders the KL is
negligible by construction (weight 1e-6) and the figure measures reconstruction
only, so we omit it rather than report a misleading likelihood.

Usage:
    python eval_recon_metrics.py --dataset CelebA
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math

import pandas as pd
import torch
import torch.nn.functional as F

from train_stage1_fast import image_cache, to_float

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def psnr(x, y):
    """x, y in [-1, 1]; PSNR on the [0, 1] scale."""
    mse = ((x - y) / 2).pow(2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp(min=1e-12))).mean().item()


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2, win=11, sigma=1.5):
    """Gaussian-window SSIM on [0,1], averaged over channels."""
    x = (x + 1) / 2
    y = (y + 1) / 2
    g = torch.arange(win, dtype=torch.float32, device=x.device) - win // 2
    g = torch.exp(-g.pow(2) / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] @ g[None, :]).expand(x.shape[1], 1, win, win).contiguous()
    pad = win // 2

    def flt(z):
        return F.conv2d(F.pad(z, [pad] * 4, mode='reflect'), k, groups=z.shape[1])

    mx, my = flt(x), flt(y)
    mxx, myy, mxy = flt(x * x), flt(y * y), flt(x * y)
    vx, vy = mxx - mx * mx, myy - my * my
    vxy = mxy - mx * my
    s = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))
    return s.flatten(1).mean(1).mean().item()


def disc_gauss_bpd(x, rec, kl_nats=0.0):
    """Discretised-Gaussian NLL bound in bits/dim.

    sigma is set to the RMSE (the calibrated ML value), and each pixel's
    probability is the Gaussian mass over its 1/255-wide bin.
    """
    npix = x[0].numel()
    mse = (x - rec).pow(2).mean()
    sigma = mse.sqrt().clamp(min=1e-4)
    half_bin = 1.0 / 255.0            # x lives in [-1,1]: bin width 2/255
    d = torch.distributions.Normal(rec, sigma)
    logp = torch.log((d.cdf(x + half_bin) - d.cdf(x - half_bin)).clamp(min=1e-12))
    nll_nats = -logp.flatten(1).sum(1).mean() + kl_nats
    return (nll_nats / (npix * math.log(2))).item()


@torch.no_grad()
def evaluate(name, recon_fn, X, bs=250, n=10000, kl_fn=None):
    ps = ss = bp = 0.0
    lp = 0.0
    import lpips
    percep = lpips.LPIPS(net='alex').to(DEVICE).eval()
    steps = 0
    for i in range(0, min(n, len(X)) - bs + 1, bs):
        x = to_float(X[i:i + bs]).to(DEVICE)
        rec = recon_fn(x)
        kl = kl_fn(x) if kl_fn else 0.0
        ps += psnr(x, rec); ss += ssim(x, rec)
        lp += percep(rec, x).mean().item()
        bp += disc_gauss_bpd(x, rec, kl)
        steps += 1
    return dict(model=name, PSNR=round(ps / steps, 2), SSIM=round(ss / steps, 4),
                LPIPS=round(lp / steps, 4), bpd=round(bp / steps, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--n', type=int, default=10000)
    a = ap.parse_args()

    X = image_cache(a.dataset, 'test', a.image_size)
    rows = []

    from src.models.ldm_ae import LdmAutoencoder
    for tag, label in [(f'ldmae_{a.dataset}_z4', 'adversarial AE'),
                       (f'ldmae_{a.dataset}_z4_sigma', 'sigma-VAE')]:
        p = f'checkpoints/{tag}.pth'
        if not os.path.exists(p):
            continue
        ae = LdmAutoencoder(base=128, z_ch=4, n_res=2).to(DEVICE)
        ae.load_state_dict(torch.load(p, map_location=DEVICE)); ae.eval()
        rows.append(evaluate(label, lambda x, m=ae: m(x, sample=False)[0], X, n=a.n))
        print(rows[-1], flush=True)

    from src.models.vq_ae import VqAutoencoder
    for nc in [256, 1024]:
        p = f'checkpoints/vqae_{a.dataset}_z4_c{nc}_ema.pth'
        if not os.path.exists(p):
            continue
        ae = VqAutoencoder(base=128, z_ch=4, n_res=2, n_codes=nc,
                           quantizer='ema').to(DEVICE)
        ae.load_state_dict(torch.load(p, map_location=DEVICE)); ae.eval()
        rows.append(evaluate(f'VQ-VAE ({nc} codes)',
                             lambda x, m=ae: m(x)[0], X, n=a.n))
        print(rows[-1], flush=True)

    from src.models.nvae_proper import ProperNVAE
    cfgs = [('s3g2z8', 8, 2, 'batch', 'NVAE over-complete (BN)'),
            ('s3g2z8_gn', 8, 2, 'group', 'NVAE over-complete (GN)'),
            ('s3g400z1', 1, [4, 0, 0], 'batch', 'NVAE matched (BN)'),
            ('s3g400z1_sched0.3', 1, [4, 0, 0], 'batch',
             'NVAE matched + sched samp')]
    for suf, zc, gl, norm, label in cfgs:
        p = f'checkpoints/nvaeP_{a.dataset}_{suf}.pth'
        if not os.path.exists(p):
            continue
        m = ProperNVAE(base=64, z_ch=zc, scales=3, groups_per_scale=gl,
                       img=a.image_size, norm=norm).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE)); m.eval()
        rows.append(evaluate(label, lambda x, mm=m: mm(x)[0], X, n=a.n,
                             kl_fn=lambda x, mm=m: sum(k.mean() for k in mm(x)[1])))
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    out = f'results/{a.dataset}/recon_metrics.csv'
    df.to_csv(out, index=False)
    print(f"\n=== RECONSTRUCTION METRICS ({a.dataset}, {a.n} held-out) ===")
    print(df.to_string(index=False))
    print("\nbpd = discretised-Gaussian NLL bound; includes the KL only for the "
          "NVAE rows, where an ELBO exists.")


if __name__ == '__main__':
    main()
