"""Stage 1 (LDM-style): train a sharp autoencoder with LPIPS + adversarial loss.

Why this stage first: the measured ceiling of the existing compressors -- FID
when decoding TRUE encoder latents -- is 5.9, and it is set entirely by the
decoder. A pure-MSE decoder is blurry no matter how good the prior is, so this
is the only stage that raises the ceiling rather than closing distance to it.

Success criteria (both reported at the end):
  * ceiling FID (decode true latents) below the NVAE's 5.9
  * Gaussian-prior FID improves over the NVAE stack

Usage:
    python train_ldm_ae.py --epochs 40
    python train_ldm_ae.py --epochs 40 --resume
    python train_ldm_ae.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch
import torch.nn.functional as F

from src.models.ldm_ae import (LdmAutoencoder, PatchDiscriminator,
                               hinge_d_loss, adaptive_weight)
from train_stage1_fast import image_cache, to_float

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def make_tag(a):
    sv = "_sigma" if a.sigma_vae else ""
    return a.tag or (f"ldmae_{a.dataset}_z{a.z_ch}{sv}" if a.image_size == 32
                     else f"ldmae_{a.dataset}{a.image_size}_z{a.z_ch}{sv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--z_ch', type=int, default=4)     # f=4 -> (S/4)^2 * z_ch dims
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--lpips_net', default='vgg', choices=['vgg', 'alex'],
                    help="alex is ~5x cheaper; VGG dominates cost at 64x64")
    ap.add_argument('--amp', action='store_true',
                    help='mixed precision; LPIPS-VGG dominates cost at 64x64')
    ap.add_argument('--base', type=int, default=128)
    ap.add_argument('--n_res', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--kl_weight', type=float, default=1e-6)
    ap.add_argument('--sigma_vae', action='store_true',
                    help="sigma-VAE: calibrate decoder variance analytically "
                         "(sigma^2 = MSE) instead of hand-tuning kl_weight")
    ap.add_argument('--perc_weight', type=float, default=1.0)
    ap.add_argument('--disc_weight', type=float, default=0.5)
    ap.add_argument('--disc_start_ep', type=int, default=6)
    ap.add_argument('--disc_lr_mult', type=float, default=2.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    tag = make_tag(a)
    Xtr = image_cache(a.dataset, 'train', a.image_size).to(DEVICE)
    Xva = image_cache(a.dataset, 'test', a.image_size).to(DEVICE)
    print(f"images train={tuple(Xtr.shape)} val={tuple(Xva.shape)}")
    SP = a.image_size // 4
    npix = a.image_size * a.image_size * 3
    print(f"latent: {SP}x{SP}x{a.z_ch} = {SP*SP*a.z_ch} dims "
          f"({SP*SP*a.z_ch/npix:.3f}x the {a.image_size}x{a.image_size}x3 image)")

    ae = LdmAutoencoder(base=a.base, z_ch=a.z_ch, n_res=a.n_res).to(DEVICE)
    disc = PatchDiscriminator().to(DEVICE)
    print(f"AE params {sum(p.numel() for p in ae.parameters())/1e6:.2f}M | "
          f"D params {sum(p.numel() for p in disc.parameters())/1e6:.2f}M")

    import lpips
    percep = lpips.LPIPS(net=a.lpips_net).to(DEVICE).eval()
    for p in percep.parameters():
        p.requires_grad_(False)

    opt_ae = torch.optim.Adam(ae.parameters(), lr=a.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr * a.disc_lr_mult, betas=(0.5, 0.9))

    ck = f"checkpoints/{tag}.pth"
    sp = f"checkpoints/{tag}_train_state.pth"
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        ae.load_state_dict(st['ae'])
        opt_ae.load_state_dict(st['opt_ae'])
        try:
            disc.load_state_dict(st['disc']); opt_d.load_state_dict(st['opt_d'])
        except Exception:
            print("  (discriminator architecture changed -- reinitializing D)")
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed at epoch {start} (best val nll {best:.4f})")

    n, bs = len(Xtr), a.bs
    if not a.eval_only:
        for ep in range(start, a.epochs):
            ae.train(); disc.train()
            perm = torch.randperm(n, device=DEVICE)
            agg = dict(rec=0., perc=0., g=0., d=0., w=0.)
            steps = 0
            use_gan = ep >= a.disc_start_ep
            t0 = time.time()
            for i in range(0, n - bs + 1, bs):
                x = to_float(Xtr[perm[i:i + bs]])
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                    x_rec, mu, logvar, _ = ae(x)
                    rec = (x - x_rec).abs().mean()
                    perc = percep(x_rec, x).mean()
                nll = (rec + a.perc_weight * perc).float()
                kl = (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                      ).sum(dim=(1, 2, 3)).mean().float()
                if a.sigma_vae:
                    # Rybkin et al.: the optimal observation variance is the MSE,
                    # so the calibrated Gaussian NLL is (D/2) log(MSE) up to a
                    # constant. This puts reconstruction and KL in the same units
                    # and removes kl_weight as a free parameter.
                    mse = ((x - x_rec.float()) ** 2).mean()
                    npix = x[0].numel()
                    loss_ae = 0.5 * npix * torch.log(mse + 1e-8) + kl                               + a.perc_weight * perc.float()
                else:
                    loss_ae = nll + a.kl_weight * kl

                if use_gan:
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                        g_loss = -disc(x_rec).mean()
                    g_loss = g_loss.float()
                    try:
                        w = adaptive_weight(nll, g_loss, ae.last_layer())
                    except RuntimeError:
                        w = torch.tensor(0.0, device=DEVICE)
                    loss_ae = loss_ae + a.disc_weight * w * g_loss
                    agg['g'] += float(g_loss); agg['w'] += float(w)

                opt_ae.zero_grad(set_to_none=True)
                loss_ae.backward()
                torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                opt_ae.step()

                if use_gan:
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                        d_loss = hinge_d_loss(disc(x), disc(x_rec.detach())).float()
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                    opt_d.step()
                    agg['d'] += float(d_loss)

                agg['rec'] += float(rec); agg['perc'] += float(perc)
                steps += 1; gstep += 1

            ae.eval()
            with torch.no_grad():
                vr = vp = vs = 0
                for j in range(0, min(len(Xva), 5120) - bs + 1, bs):
                    x = to_float(Xva[j:j + bs])
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                        xr, _, _, _ = ae(x, sample=False)
                        vr += float((x - xr).abs().mean())
                        vp += float(percep(xr, x).mean())
                    vs += 1
                vr /= vs; vp /= vs
            val = vr + a.perc_weight * vp
            star = ''
            if val < best:
                best = val; torch.save(ae.state_dict(), ck); star = '  *'
            print(f"  ep {ep+1:3d}/{a.epochs}  L1 {agg['rec']/steps:.4f}  "
                  f"LPIPS {agg['perc']/steps:.4f}  g {agg['g']/max(steps,1):+.3f}  "
                  f"d {agg['d']/max(steps,1):.3f} w {agg['w']/max(steps,1):.3f} | "
                  f"val L1 {vr:.4f} LPIPS {vp:.4f}"
                  f"{star}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(ae=ae.state_dict(), disc=disc.state_dict(),
                            opt_ae=opt_ae.state_dict(), opt_d=opt_d.state_dict(),
                            epoch=ep + 1, gstep=gstep, best=best), sp)

    # ---------------- evaluation ----------------
    ae.load_state_dict(torch.load(ck, map_location=DEVICE)); ae.eval()
    from fid_standard import StandardFid, run

    with torch.no_grad():
        mus = []
        for j in range(0, len(Xtr), 512):
            mu, _ = ae.encode(to_float(Xtr[j:j + 512]))
            mus.append(mu.cpu())
        Zmu = torch.cat(mus)
    scale = 1.0 / Zmu.std()
    print(f"\nlatent stats: std {Zmu.std():.4f} -> LDM scale factor {scale:.4f}")
    torch.save({'z': Zmu, 'scale': float(scale)},
               f"checkpoints/_ldmae_latents_{a.dataset}"
               f"{'' if a.image_size == 32 else a.image_size}_{a.z_ch}"
               f"{'_sigma' if a.sigma_vae else ''}.pt")

    Zt = Zmu.to(DEVICE)
    mean_z = Zt.mean(0, keepdim=True)
    std_z = Zt.std(0, keepdim=True)

    fid = StandardFid(n_real=min(50000, len(Xtr)), dataset=a.dataset,
                      image_size=a.image_size)
    cfg = []

    def ceiling(b):
        idx = torch.randint(0, len(Zt), (b,), device=DEVICE)
        return ae.decode(Zt[idx])
    cfg.append((f'ldm_ae_{a.dataset}', f'TRUE encoder latents [ceiling] z{a.z_ch}', 0, ceiling))

    def gauss(sg):
        def f(b):
            z = mean_z + sg * std_z * torch.randn(b, *Zt.shape[1:], device=DEVICE)
            return ae.decode(z)
        return f
    for sg in [1.0, 1.2]:
        cfg.append((f'ldm_ae_{a.dataset}', f'per-dim Gaussian prior sigma={sg} z{a.z_ch}', 0, gauss(sg)))

    print(f"\nstandard FID (50k vs 50k)\n")
    df = run(cfg, min(50000, len(Xtr)), fid,
             f"results/{a.dataset}/ldm_ae{'' if a.image_size == 32 else a.image_size}.csv",
             variant='sigma-vae' if a.sigma_vae else 'ldm')
    print("\n=== LDM-STYLE AUTOENCODER ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
