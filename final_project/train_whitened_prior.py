"""Whitened-residual prior: start at the covariance baseline, learn only the rest.

Diagnostics on the compressed NVAE's fine-latent residual
eps = (z_fine - mu_p)/sigma_p:

    cov_err(N(0,I) vs eps)          0.589     -> covariance structure is real
    full-cov Gaussian fit           FID 41.65 -> and worth ~14 FID
    after whitening, excess kurtosis +0.547   -> heavy tails REMAIN
    skewness                         0.005    -> symmetric
    covariance spectrum       top-50 = 19.2%  -> high rank (eff. rank 428/512)

So the residual is heavy-tailed and high-rank, not low-rank-correlated. A conv
net over the raw residual was chasing spatial structure that whitening shows is
not the issue -- and it learned nothing (cov_err 0.558 vs 0.589 for white noise,
at any budget up to 150 epochs).

This trains in WHITENED coordinates instead:

    u = W (eps - mu),   W = L^-1,  L L^T = Cov(eps)      [train target]
    eps = mu + L u                                        [sampling]

The key property: if the network learns u ~ N(0, I) it reproduces the
full-covariance Gaussian EXACTLY, so FID 41.65 is a floor, not a risk. Anything
it learns about the heavy tails is pure gain. An MLP is used rather than a conv
net because whitening mixes all 512 dims -- there is no spatial structure left
to exploit.

Usage:
    python train_whitened_prior.py --epochs 100
    python train_whitened_prior.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import pandas as pd
import torch

from src.models.meanflow import MeanFlow, MeanFlowNet
from train_meanflow import Ema, DEVICE
from train_joint_prior import (load_base, joint_latents, decode_joint, Z_G, Z_F)
from train_residual_fine_prior import decoder_fine_prior

TAG = 'nvae_CelebA_fine8_whitened_meanflow'
TARGET_CLIP = 90.0


def whitening(Etr, ridge=1e-4):
    mu = Etr.mean(0, keepdim=True)
    E = Etr - mu
    C = (E.T @ E) / (len(E) - 1)
    I = torch.eye(C.shape[0], device=C.device, dtype=torch.float64)
    L = torch.linalg.cholesky(C.double() + ridge * I)
    W = torch.linalg.inv(L)
    return mu, L.float(), W.float()


def build(width, blocks):
    net = MeanFlowNet(z_dim=Z_F, model_dim=width, n_blocks=blocks, cond_dim=Z_G)
    # target is already whitened: zero mean, unit covariance
    return MeanFlow(net, D=Z_F, variant='base',
                    data_mean=torch.zeros(Z_F), data_std=torch.ones(Z_F),
                    target_clip=TARGET_CLIP).to(DEVICE)


def nongauss(u):
    """Excess kurtosis of a standardized sample -- the thing we are trying to learn."""
    z = (u - u.mean(0)) / u.std(0).clamp(min=1e-6)
    return ((z ** 4).mean() - 3).item()


@torch.no_grad()
def eval_u(mf, Uva, Zva, steps=1, n=4096):
    idx = torch.randperm(len(Uva), device=DEVICE)[:n]
    g = mf.sample(0, steps=steps, cond=Zva[idx][:, :Z_G])
    r = Uva[idx]
    return dict(d_std=(g.std(0) - r.std(0)).abs().mean().item(),
                kurt=nongauss(g), kurt_true=nongauss(r))


def train(Utr, Uva, Ztr, Zva, a):
    mf = build(a.width, a.blocks)
    ema = Ema(mf, a.ema)
    opt = torch.optim.AdamW(mf.parameters(), lr=a.lr, betas=(0.9, 0.99))
    n, bs = len(Utr), a.bs
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
        print(f"  resumed at epoch {start} (best kurt gap {best:.4f})")

    t0 = time.time()
    for ep in range(start, a.epochs):
        mf.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = steps = 0
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]
            for g in opt.param_groups:
                g['lr'] = a.lr * lr_at(gstep)
            loss = mf.compute_loss(Utr[j], cond=Ztr[j][:, :Z_G])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mf.parameters(), 0.5)
            opt.step(); ema.update(mf)
            tot += loss.item(); steps += 1; gstep += 1

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            m = eval_u(ema.shadow, Uva, Zva)
            # selection target: reproduce the true excess kurtosis, keep unit scale
            gap = abs(m['kurt'] - m['kurt_true']) + 3.0 * m['d_std']
            star = ''
            if gap < best:
                best = gap; torch.save(ema.shadow.state_dict(), ck); star = '  *'
            print(f"  [white] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:8.3f}  "
                  f"dstd {m['d_std']:.4f}  kurt {m['kurt']:+.3f} "
                  f"(true {m['kurt_true']:+.3f})  gap {gap:.4f}{star}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=mf.state_dict(), ema=ema.shadow.state_dict(),
                            opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                            best=best), sp)
    return ck, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    ap.add_argument('--fake', type=int, default=50000)
    a = ap.parse_args()

    base = load_base()
    Ztr = joint_latents(base, 'train').to(DEVICE)
    Zva = joint_latents(base, 'test').to(DEVICE)
    d = torch.load('checkpoints/_resid_fine.pt')
    Etr, Eva = d['train'].to(DEVICE), d['val'].to(DEVICE)

    mu, L, W = whitening(Etr)
    Utr, Uva = (Etr - mu) @ W.T, (Eva - mu) @ W.T
    print(f"whitened target: std {Utr.std():.4f}  excess kurtosis {nongauss(Uva):+.4f} "
          f"(this is what the network must learn; 0 would mean nothing left)")

    if not a.eval_only:
        print(f"\n=== whitened-space MeanFlow, floor = full-cov Gaussian ===", flush=True)
        train(Utr, Uva, Ztr, Zva, a)

    from fid_standard import StandardFid, run
    fid = StandardFid(n_real=50000)
    cfg = []

    def mk_gauss(sg):
        def f(b):
            zg = sg * torch.randn(b, Z_G, device=DEVICE)
            mp, sp = decoder_fine_prior(base, zg)
            u = torch.randn(b, Z_F, device=DEVICE)
            return decode_joint(base, torch.cat([zg, mp + sp * (mu + u @ L.T)], 1))
        return f

    for sg in [1.0, 1.4]:
        cfg.append(('nvae_fine8', f'z_g~N(0,{sg}) + FULL-COV residual [floor]', 0, mk_gauss(sg)))

    ck = f"checkpoints/{TAG}.pth"
    if os.path.exists(ck):
        mf = build(a.width, a.blocks)
        mf.load_state_dict(torch.load(ck, map_location=DEVICE)); mf.eval()
        m = eval_u(mf, Uva, Zva)
        print(f"  learned u: dstd {m['d_std']:.4f}  kurt {m['kurt']:+.3f} "
              f"vs true {m['kurt_true']:+.3f}")

        def mk_nn(sg, st):
            def f(b):
                zg = sg * torch.randn(b, Z_G, device=DEVICE)
                mp, sp = decoder_fine_prior(base, zg)
                u = mf.sample(0, steps=st, cond=zg)
                return decode_joint(base, torch.cat([zg, mp + sp * (mu + u @ L.T)], 1))
            return f
        for sg in [1.0, 1.4]:
            for st in [1, 2]:
                cfg.append(('nvae_fine8',
                            f'z_g~N(0,{sg}) + WHITENED MeanFlow', st, mk_nn(sg, st)))

    print(f"\nstandard FID: {a.fake} vs 50000\n")
    df = run(cfg, a.fake, fid, 'results/CelebA/whitened_prior.csv')
    print("\n=== WHITENED-SPACE PRIOR vs COVARIANCE FLOOR ===")
    print(df.to_string(index=False))
