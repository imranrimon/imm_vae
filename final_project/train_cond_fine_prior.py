"""Conditional Stage-2 prior: p(z_fine | z_global), the form a hierarchical VAE needs.

Measured on the compressed NVAE (z_fine_ch=8), the generative gap decomposes as:

    decode TRUE (z_global, z_fine) from the encoder .............. FID   6.1
    z_global ~ N(0,I), z_fine ~ decoder's Gaussian p(z_fine|h) ... FID  55.1
    unconditional joint MeanFlow over [z_global, z_fine] ......... FID 216.4

So essentially the whole gap sits in p(z_fine | z_global) -- the decoder's own
diagonal Gaussian -- and NOT in p(z_global), which is already ~N(0, I). An
unconditional joint prior is worse than useless because it breaks the coupling
between the two levels.

This trains a MeanFlow prior over z_fine *conditioned on* z_global, replacing
the decoder's Gaussian while preserving the hierarchy.

Usage:
    python train_cond_fine_prior.py --epochs 30
    python train_cond_fine_prior.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import pandas as pd
import torch

from src.models.meanflow import MeanFlow, MeanFlowNet
from train_meanflow import Ema, rbf_mmd, DEVICE
from train_joint_prior import (load_base, joint_latents, decode_joint,
                               Z_G, Z_F, FINE_CH, FINE_SP)

TAG = 'nvae_CelebA_fine8_condfine_meanflow'


def build(mean, std, width):
    net = MeanFlowNet(z_dim=Z_F, model_dim=width, n_blocks=6, cond_dim=Z_G)
    return MeanFlow(net, D=Z_F, variant='base',
                    data_mean=mean, data_std=std).to(DEVICE)


def train(Ztr, Zva, a):
    xf = Ztr[:, Z_G:]
    mean, std = xf.mean(0).to(DEVICE), xf.std(0).to(DEVICE)
    mf = build(mean, std, a.width)
    ema = Ema(mf, a.ema)
    opt = torch.optim.AdamW(mf.parameters(), lr=a.lr, betas=(0.9, 0.99))

    Ztr, Zva = Ztr.to(DEVICE), Zva.to(DEVICE)
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
        print(f"  resumed at epoch {start} (best mmd {best:.5f})")

    t0 = time.time()
    for ep in range(start, a.epochs):
        mf.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = steps = 0
        for i in range(0, n - bs + 1, bs):
            v = Ztr[perm[i:i + bs]]
            for g in opt.param_groups:
                g['lr'] = a.lr * lr_at(gstep)
            loss = mf.compute_loss(v[:, Z_G:], cond=v[:, :Z_G])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mf.parameters(), 1.0)
            opt.step(); ema.update(mf)
            tot += loss.item(); steps += 1; gstep += 1

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                v = Zva[torch.randperm(len(Zva), device=DEVICE)[:4096]]
                g = ema.shadow.sample(0, steps=1, cond=v[:, :Z_G])
            r = v[:, Z_G:]
            m = rbf_mmd(g, r)
            ds = (g.std(0) - r.std(0)).abs().mean().item()
            star = ''
            if m < best:
                best = m; torch.save(ema.shadow.state_dict(), ck); star = '  *'
            print(f"  [{TAG}] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:9.3f}  "
                  f"dstd {ds:.4f}  mmd {m:.5f}{star}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=mf.state_dict(), ema=ema.shadow.state_dict(),
                            opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                            best=best), sp)
    return ck, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    ap.add_argument('--fake', type=int, default=5000)
    ap.add_argument('--real', type=int, default=10000)
    a = ap.parse_args()

    base = load_base()
    Ztr, Zva = joint_latents(base, 'train'), joint_latents(base, 'test')

    if not a.eval_only:
        print(f"\n=== training CONDITIONAL p(z_fine | z_global), D={Z_F}, cond={Z_G} ===",
              flush=True)
        train(Ztr, Zva, a)

    from ab_sampler_eval import Fid, gen_images
    fid = Fid()
    rows = []

    def add(name, nfe, make_imgs):
        torch.manual_seed(0)
        imgs, done = [], 0
        while done < a.fake:
            b = min(500, a.fake - done)
            with torch.no_grad():
                imgs.append(make_imgs(b).cpu())
            done += b
        f = fid.score(torch.cat(imgs)[:a.fake], a.real)
        rows.append(dict(Prior=name, NFE=nfe, FID=round(f, 3)))
        print(f"  {name:40s} NFE={nfe:>3}  FID={f:7.3f}", flush=True)

    Zva = Zva.to(DEVICE)
    # upper bound: real encoder latents through the decoder
    add('TRUE encoder (z_global, z_fine)', 0,
        lambda b: decode_joint(base, Zva[torch.randint(0, len(Zva), (b,), device=DEVICE)]))
    # original Stage-2 setup: Gaussian z_global, decoder's own p(z_fine|h)
    for sg in [1.0, 1.5]:
        add(f'N(0,{sg}) z_global + decoder p(z_fine)', 0,
            lambda b, s=sg: (lambda o: o[0] if isinstance(o, tuple) else o)(
                base.decoder(s * torch.randn(b, Z_G, device=DEVICE), z_fine=None)))

    ck = f"checkpoints/{TAG}.pth"
    if os.path.exists(ck):
        xf = Ztr[:, Z_G:]
        mf = build(xf.mean(0).to(DEVICE), xf.std(0).to(DEVICE), a.width)
        mf.load_state_dict(torch.load(ck, map_location=DEVICE)); mf.eval()
        for st in [1, 2, 4]:
            def mk(b, st=st):
                zg = torch.randn(b, Z_G, device=DEVICE)
                zf = mf.sample(0, steps=st, cond=zg)
                return decode_joint(base, torch.cat([zg, zf], dim=1))
            add('N(0,1) z_global + COND MeanFlow p(z_f|z_g)', st, mk)

    df = pd.DataFrame(rows)
    out = 'results/CelebA/cond_fine_prior.csv'
    os.makedirs('results/CelebA', exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== CONDITIONAL z_fine PRIOR (compressed NVAE) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")
