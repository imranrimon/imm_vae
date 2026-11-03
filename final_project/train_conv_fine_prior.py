"""Spatial (conv) MeanFlow prior over z_fine, conditioned on z_global.

The flat-MLP prior matched every marginal statistic of z_fine yet produced
FID 218 (true latents: 7.1). z_fine is an 8x8x8 tensor, so this uses a conv
ResNet over the grid instead -- see src/models/conv_meanflow.py.

Model selection uses a COVARIANCE error, not RBF-MMD: the previous run showed
MMD 0.012 (healthy-looking) for a prior that was in fact catastrophic, because
median-heuristic RBF-MMD on 512 dims is insensitive to spatial covariance.

    cov_err = ||Cov(gen) - Cov(true)||_F / ||Cov(true)||_F

Usage:
    python train_conv_fine_prior.py --epochs 30
    python train_conv_fine_prior.py --eval-only
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

TAG = 'nvae_CelebA_fine8_convfine_meanflow'


def build(mean, std, width, blocks):
    net = ConvMeanFlowNet(channels=FINE_CH, spatial=FINE_SP, width=width,
                          n_blocks=blocks, cond_dim=Z_G)
    return MeanFlow(net, D=Z_F, variant='base',
                    data_mean=mean, data_std=std).to(DEVICE)


def cov_err(g, r):
    """Relative Frobenius error of the latent covariance -- catches the spatial
    structure that per-dim moments and RBF-MMD both miss."""
    g = g - g.mean(0, keepdim=True)
    r = r - r.mean(0, keepdim=True)
    Cg = (g.T @ g) / (len(g) - 1)
    Cr = (r.T @ r) / (len(r) - 1)
    return ((Cg - Cr).norm() / Cr.norm()).item()


@torch.no_grad()
def evaluate_latents(mf, Zva, steps=1, n=4096):
    v = Zva[torch.randperm(len(Zva), device=DEVICE)[:n]]
    g = mf.sample(0, steps=steps, cond=v[:, :Z_G])
    r = v[:, Z_G:]
    return dict(d_std=(g.std(0) - r.std(0)).abs().mean().item(),
                mmd=rbf_mmd(g, r),
                cov=cov_err(g, r))


def train(Ztr, Zva, a):
    xf = Ztr[:, Z_G:]
    mf = build(xf.mean(0).to(DEVICE), xf.std(0).to(DEVICE), a.width, a.blocks)
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
        print(f"  resumed at epoch {start} (best cov_err {best:.5f})")

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
            m = evaluate_latents(ema.shadow, Zva)
            star = ''
            if m['cov'] < best:
                best = m['cov']; torch.save(ema.shadow.state_dict(), ck); star = '  *'
            print(f"  [conv] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:9.3f}  "
                  f"dstd {m['d_std']:.4f}  mmd {m['mmd']:.5f}  "
                  f"cov_err {m['cov']:.4f}{star}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=mf.state_dict(), ema=ema.shadow.state_dict(),
                            opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                            best=best), sp)
    return ck, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
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

    if not a.eval_only:
        print(f"\n=== conv MeanFlow p(z_fine | z_global), {FINE_CH}x{FINE_SP}x{FINE_SP} ===",
              flush=True)
        train(Ztr, Zva, a)

    from ab_sampler_eval import Fid
    fid = Fid()
    Zva = Zva.to(DEVICE)
    rows = []

    def add(name, nfe, mk, extra=None):
        torch.manual_seed(0)
        imgs, done = [], 0
        while done < a.fake:
            b = min(500, a.fake - done)
            with torch.no_grad():
                imgs.append(mk(b).cpu())
            done += b
        f = fid.score(torch.cat(imgs)[:a.fake], a.real)
        row = dict(Prior=name, NFE=nfe, FID=round(f, 3))
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {name:44s} NFE={nfe:>3}  FID={f:7.3f}", flush=True)

    add('TRUE encoder (z_global, z_fine)', 0,
        lambda b: decode_joint(base, Zva[torch.randint(0, len(Zva), (b,), device=DEVICE)]))
    for sg in [1.0, 1.5]:
        add(f'N(0,{sg}) z_global + decoder p(z_fine|h)', 0,
            lambda b, s=sg: (lambda o: o[0] if isinstance(o, tuple) else o)(
                base.decoder(s * torch.randn(b, Z_G, device=DEVICE), z_fine=None)))

    ck = f"checkpoints/{TAG}.pth"
    if os.path.exists(ck):
        xf = Ztr[:, Z_G:]
        mf = build(xf.mean(0).to(DEVICE), xf.std(0).to(DEVICE), a.width, a.blocks)
        mf.load_state_dict(torch.load(ck, map_location=DEVICE)); mf.eval()
        m = evaluate_latents(mf, Zva)
        print(f"  latent fidelity: dstd {m['d_std']:.4f}  mmd {m['mmd']:.5f}  "
              f"cov_err {m['cov']:.4f}")
        for st in [1, 2, 4]:
            def mk(b, st=st):
                zg = torch.randn(b, Z_G, device=DEVICE)
                zf = mf.sample(0, steps=st, cond=zg)
                return decode_joint(base, torch.cat([zg, zf], dim=1))
            add('N(0,1) z_global + CONV MeanFlow p(z_f|z_g)', st, mk,
                {k: round(v, 5) for k, v in m.items()})

    df = pd.DataFrame(rows)
    out = 'results/CelebA/conv_fine_prior.csv'
    df.to_csv(out, index=False)
    print("\n=== SPATIAL z_fine PRIOR (compressed NVAE) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")
