"""Train the proper NVAE and place it on the diagnostic axis.

NVAE's prior is conditional and trained jointly with the encoder, so ancestral
sampling from the model already IS a learned hierarchical prior. The rung this
adds to the ladder therefore tests a specific prediction: if a compressor closes
the prior--posterior gap architecturally, a Stage-2 prior should buy very little
on top of it.

Reported on the same three quantities as every other compressor:
  ceiling      decode posterior samples from held-out images
  free         ancestral sampling from the model's own prior (t=1), plus a
               temperature sweep, which is NVAE's analogue of our Gaussian control
  learned      a Stage-2 prior over the topmost group, the only group that is
               unconditional

Usage:
    python train_nvae_proper.py --epochs 40
    python train_nvae_proper.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time

import torch
import torch.nn.functional as F

from src.models.nvae_proper import ProperNVAE
from train_stage1_fast import image_cache, to_float
from fid_standard import StandardFid, run

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--z_ch', type=int, default=8)
    ap.add_argument('--scales', type=int, default=3)
    ap.add_argument('--gps', default='2',
                    help='groups per scale: int, or comma list e.g. 4,0,0')
    ap.add_argument('--norm', default='batch', choices=['batch', 'group'])
    ap.add_argument('--no-bn-recal', dest='bn_recal', action='store_false')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--kl_warmup', type=int, default=10)
    ap.add_argument('--free_bits', type=float, default=0.05)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    a = ap.parse_args()
    gps = ([int(v) for v in a.gps.split(',')] if ',' in a.gps
           else int(a.gps))
    gtag = a.gps.replace(',', '')

    Xtr = image_cache(a.dataset, 'train', a.image_size).to(DEVICE)
    Xva = image_cache(a.dataset, 'test', a.image_size).to(DEVICE)

    m = ProperNVAE(base=a.base, z_ch=a.z_ch, scales=a.scales,
                   groups_per_scale=gps, img=a.image_size, norm=a.norm).to(DEVICE)
    npix = a.image_size * a.image_size * 3
    print(f"proper NVAE: {m.n_groups} latent groups over {a.scales} scales, "
          f"{m.latent_dims()} latent dims ({m.latent_dims()/npix:.2f}x the image), "
          f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M params")

    nm = "" if a.norm == "batch" else "_gn"
    tag = f"nvaeP_{a.dataset}_s{a.scales}g{gtag}z{a.z_ch}{nm}"
    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_state.pth"
    opt = torch.optim.Adamax(m.parameters(), lr=a.lr, weight_decay=3e-4)
    start, best = 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        m.load_state_dict(st['m']); opt.load_state_dict(st['opt'])
        start, best = st['epoch'], st['best']
        print(f"  resumed at epoch {start} (best {best:.4f})")

    n, bs = len(Xtr), a.bs
    with torch.no_grad():                      # dims per latent group
        _, _kl0 = m(to_float(Xtr[:2]))
    kl_dims = []
    r0 = a.image_size // 2 ** (a.scales - 1)
    for s_ in range(a.scales):
        for _ in range(m.gps_list[s_]):
            kl_dims.append(a.z_ch * r0 * r0)
        r0 *= 2
    if not a.eval_only:
        for ep in range(start, a.epochs):
            beta = min(1.0, (ep + 1) / a.kl_warmup) if a.kl_warmup else 1.0
            m.train()
            perm = torch.randperm(n, device=DEVICE)
            tr = tk = steps = 0
            t0 = time.time()
            for i in range(0, n - bs + 1, bs):
                x = to_float(Xtr[perm[i:i + bs]])
                rec, kls = m(x)
                # Calibrated Gaussian likelihood (sigma^2 = MSE), as in the
                # sigma-VAE rung. Plain summed MSE implies sigma^2 = 0.5, which
                # over-weights the KL by orders of magnitude and drives the
                # hierarchy to collapse as beta ramps -- observed directly as
                # val MSE255 rising 240 -> 342 over the warm-up.
                mse = F.mse_loss(rec, x)
                r = 0.5 * npix * torch.log(mse + 1e-8)
                # Free bits applied PER DIMENSION, not per group: a group-level
                # floor of 0.05 nats never binds against a summed KL in the tens.
                kl = 0
                for k, dims in zip(kls, kl_dims):
                    kl = kl + torch.clamp(k.mean() / dims, min=a.free_bits) * dims
                loss = r + beta * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 200.0)
                opt.step()
                tr += float(r); tk += float(kl); steps += 1

            m.eval()
            with torch.no_grad():
                vr = vs = 0
                for j in range(0, min(len(Xva), 5120) - bs + 1, bs):
                    x = to_float(Xva[j:j + bs])
                    rec, _ = m(x)
                    vr += float(F.mse_loss(rec, x, reduction='sum') / x.shape[0])
                    vs += 1
                vr /= vs
            mse255 = vr / npix * (127.5 ** 2)
            star = ''
            if vr < best:
                best = vr; torch.save(m.state_dict(), ck); star = '  *'
            print(f"  ep {ep+1:3d}/{a.epochs}  beta {beta:.2f}  recon {tr/steps:8.2f}  "
                  f"KL {tk/steps:8.2f} | val MSE255 {mse255:7.3f}{star}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(m=m.state_dict(), opt=opt.state_dict(),
                            epoch=ep + 1, best=best), sp)

    m.load_state_dict(torch.load(ck, map_location=DEVICE)); m.eval()

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    Xva_d = Xva

    def recon(b):
        idx = torch.randint(0, len(Xva_d), (b,), device=DEVICE)
        with torch.no_grad():
            r, _ = m(to_float(Xva_d[idx]))
        return r

    cfg = [('nvae_proper', 'posterior reconstructions [ceiling]', 0, recon)]
    for tt in [0.8, 1.0]:
        cfg.append(('nvae_proper', f'ancestral sampling t={tt}', 0,
                    lambda b, x=tt: m.sample(b, DEVICE, t=x)))
    variant = f's{a.scales}g{gtag}z{a.z_ch}{nm}'
    df = run(cfg, 50000, fid, f'results/{a.dataset}/nvae_proper.csv',
             variant=variant)

    if a.bn_recal and a.norm == 'batch':
        import copy
        cfg2 = []
        for tt in [0.8, 1.0]:
            mr = copy.deepcopy(m).recalibrate_bn(n=20000, bs=250,
                                                 device=DEVICE, t=tt)
            cfg2.append(('nvae_proper', f'ancestral t={tt} + BN recal', 0,
                         lambda b, mm=mr, x=tt: mm.sample(b, DEVICE, t=x)))
        df = run(cfg2, 50000, fid, f'results/{a.dataset}/nvae_proper.csv',
                 variant=variant)
    print("\n=== PROPER NVAE ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
