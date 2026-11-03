"""Residual z_fine prior: model what the decoder's Gaussian gets WRONG.

The decoder already carries a conditional prior p(z_fine | h(z_global)) with
learned mu_p and sigma_p. Measured on the compressed NVAE, mu_p alone has std
17.5 out of z_fine's total std 20.4 -- most of the signal is already predicted.
Modelling z_fine from scratch (flat MLP or conv, unconditional or conditioned on
z_global) threw that away and produced FID ~220 against 55.1 for the decoder's
own Gaussian.

So model the standardized residual instead:

    eps = (z_fine - mu_p(h)) / sigma_p(h)          ~ unit scale by construction
    z_fine = mu_p(h) + sigma_p(h) * eps_hat        at sampling time

If the prior learns eps ~ N(0, I) it exactly reproduces the decoder's Gaussian
(FID 55.1), so that is the floor, not a risk. Anything better is genuine gain.
The decoder-sensitivity sweep says a prior accurate to 40% relative error would
already reach FID ~28, and 10% would reach ~7.8.

Usage:
    python train_residual_fine_prior.py --epochs 30
    python train_residual_fine_prior.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import pandas as pd
import torch

from src.models.conv_meanflow import ConvMeanFlowNet
from src.models.meanflow import MeanFlow
from train_meanflow import Ema, rbf_mmd, DEVICE
from train_joint_prior import (load_base, joint_latents, decode_joint,
                               Z_G, Z_F, FINE_CH, FINE_SP)
from train_conv_fine_prior import cov_err

TAG = 'nvae_CelebA_fine8_residfine_meanflow'
# ||target|| budget: eps is unit-scale over 512 dims, so a clean target has
# norm ~sqrt(512)=22.6; allow 4x that before rescaling.
TARGET_CLIP = 90.0


@torch.no_grad()
def decoder_fine_prior(base, z_g):
    """mu_p, sigma_p of the decoder's own p(z_fine | h(z_global))."""
    d = base.decoder
    h = d.from_global(z_g).view(-1, d.base_channels * 4, d.spatial_dim, d.spatial_dim)
    h = d.up3(d.up4(h))
    st = d.prior_fine_stats(h)
    mu_p, lv_p = st.chunk(2, dim=1)
    return mu_p.flatten(1), torch.exp(0.5 * torch.clamp(lv_p, -10., 10.)).flatten(1)


@torch.no_grad()
def residuals(base, Z):
    """eps = (z_fine - mu_p) / sigma_p for a joint-latent tensor."""
    out = []
    for i in range(0, len(Z), 2048):
        v = Z[i:i + 2048].to(DEVICE)
        mu_p, sg_p = decoder_fine_prior(base, v[:, :Z_G])
        out.append(((v[:, Z_G:] - mu_p) / (sg_p + 1e-6)).cpu())
    return torch.cat(out)


def build(mean, std, width, blocks):
    net = ConvMeanFlowNet(channels=FINE_CH, spatial=FINE_SP, width=width,
                          n_blocks=blocks, cond_dim=Z_G)
    return MeanFlow(net, D=Z_F, variant='base', data_mean=mean, data_std=std,
                    target_clip=TARGET_CLIP).to(DEVICE)


@torch.no_grad()
def eval_latents(mf, base, Zva, Eva, steps=1, n=4096):
    idx = torch.randperm(len(Zva), device=DEVICE)[:n]
    zg = Zva[idx][:, :Z_G]
    g = mf.sample(0, steps=steps, cond=zg)
    r = Eva[idx]
    return dict(d_std=(g.std(0) - r.std(0)).abs().mean().item(),
                mmd=rbf_mmd(g, r), cov=cov_err(g, r))


def train(base, Ztr, Zva, Etr, Eva, a):
    mean, std = Etr.mean(0).to(DEVICE), Etr.std(0).to(DEVICE)
    mf = build(mean, std, a.width, a.blocks)
    ema = Ema(mf, a.ema)
    opt = torch.optim.AdamW(mf.parameters(), lr=a.lr, betas=(0.9, 0.99))

    Ztr, Etr = Ztr.to(DEVICE), Etr.to(DEVICE)
    n, bs = len(Ztr), a.bs
    total = (n // bs) * a.epochs
    warm = min(500, total // 10)

    def lr_at(g):
        if g < warm:
            return g / max(warm, 1)
        p = (g - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    ck, sp = f"checkpoints/{TAG}.pth", f"checkpoints/{TAG}_train_state.pth"
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        mf.load_state_dict(st['model']); ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed at epoch {start} (best cov_err {best:.5f})")

    t0 = time.time()
    for ep in range(start, a.epochs):
        mf.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = steps = 0
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]
            for g in opt.param_groups:
                g['lr'] = a.lr * lr_at(gstep)
            loss = mf.compute_loss(Etr[j], cond=Ztr[j][:, :Z_G])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mf.parameters(), 0.5)
            opt.step(); ema.update(mf)
            tot += loss.item(); steps += 1; gstep += 1

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            m = eval_latents(ema.shadow, base, Zva, Eva)
            star = ''
            if m['cov'] < best:
                best = m['cov']; torch.save(ema.shadow.state_dict(), ck); star = '  *'
            print(f"  [resid] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:8.3f}  "
                  f"dstd {m['d_std']:.4f}  mmd {m['mmd']:.5f}  cov_err {m['cov']:.4f}"
                  f"{star}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=mf.state_dict(), ema=ema.shadow.state_dict(),
                            opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                            best=best), sp)
    return ck, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    ap.add_argument('--fake', type=int, default=5000)
    ap.add_argument('--real', type=int, default=10000)
    a = ap.parse_args()

    base = load_base()
    Ztr, Zva = joint_latents(base, 'train'), joint_latents(base, 'test')

    rc = 'checkpoints/_resid_fine.pt'
    if os.path.exists(rc):
        d = torch.load(rc); Etr, Eva = d['train'], d['val']
    else:
        print("computing decoder-prior residuals...", flush=True)
        Etr, Eva = residuals(base, Ztr), residuals(base, Zva)
        torch.save({'train': Etr, 'val': Eva}, rc)
    print(f"residual eps: std {Etr.std():.4f}  mean {Etr.mean():+.4f}  "
          f"(N(0,1) would mean the decoder's Gaussian is already exact)")

    Zva_d, Eva_d = Zva.to(DEVICE), Eva.to(DEVICE)
    if not a.eval_only:
        print(f"\n=== residual conv MeanFlow over eps, cond=z_global ===", flush=True)
        train(base, Ztr, Zva_d, Etr, Eva_d, a)

    from ab_sampler_eval import Fid
    fid = Fid()
    rows = []

    def add(name, nfe, mk):
        torch.manual_seed(0)
        imgs, done = [], 0
        while done < a.fake:
            b = min(500, a.fake - done)
            with torch.no_grad():
                imgs.append(mk(b).cpu())
            done += b
        f = fid.score(torch.cat(imgs)[:a.fake], a.real)
        rows.append(dict(Prior=name, NFE=nfe, FID=round(f, 3)))
        print(f"  {name:46s} NFE={nfe:>3}  FID={f:7.3f}", flush=True)

    add('TRUE encoder (z_global, z_fine)', 0,
        lambda b: decode_joint(base, Zva_d[torch.randint(0, len(Zva_d), (b,), device=DEVICE)]))
    add('N(0,1) z_global + decoder p(z_fine|h)', 0,
        lambda b: (lambda o: o[0] if isinstance(o, tuple) else o)(
            base.decoder(torch.randn(b, Z_G, device=DEVICE), z_fine=None)))

    ck = f"checkpoints/{TAG}.pth"
    if os.path.exists(ck):
        mf = build(Etr.mean(0).to(DEVICE), Etr.std(0).to(DEVICE), a.width, a.blocks)
        mf.load_state_dict(torch.load(ck, map_location=DEVICE)); mf.eval()
        m = eval_latents(mf, base, Zva_d, Eva_d)
        print(f"  residual fidelity: dstd {m['d_std']:.4f}  mmd {m['mmd']:.5f}  "
              f"cov_err {m['cov']:.4f}")
        for st in [1, 2, 4]:
            def mk(b, st=st):
                zg = torch.randn(b, Z_G, device=DEVICE)
                mu_p, sg_p = decoder_fine_prior(base, zg)
                eps = mf.sample(0, steps=st, cond=zg)
                return decode_joint(base, torch.cat([zg, mu_p + sg_p * eps], dim=1))
            add('N(0,1) z_global + RESIDUAL MeanFlow', st, mk)

    df = pd.DataFrame(rows)
    out = 'results/CelebA/residual_fine_prior.csv'
    df.to_csv(out, index=False)
    print("\n=== RESIDUAL z_fine PRIOR ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")
