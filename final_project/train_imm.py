"""Train an IMM latent prior with the CORRECTED OT-FM convention.

The shipped IMM (convention='legacy') interpolates x_t = t*x + (1-t)*eps, so
t=1 is data, while still drawing s ~ U(tmin, t). Under that axis s < t means
"s is noisier", so the network learns a noising map. See src/models/imm_core.py.

This trainer uses convention='fixed': x_t = (1-t)*x + t*eps (t=1 is noise), so
s < t correctly means "s is closer to the data" and sampling walks t downward.

It also drops the per-iteration MMD term from src/training/stage2.py, which ran
a full 32-step sample() inside every training step (~60 extra network evals) for
a term weighted 1e-4.

Usage:
    python train_imm.py --model nvae --epochs 40
    python train_imm.py --model nvae --epochs 40 --resume
    python train_imm.py --model nvae --convention legacy --tag imm_legacy_retrain
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch

from src.models.imm_core import (IMM, OtFmEulerFmParameterization,
                                 OtFmEulerFmParameterizationFixed)
from src.models.diffusion import ResidualDenoiser
from train_meanflow import Ema, latent_dataset, rbf_mmd, DEVICE, Z_DIM


def build(convention, mean, std, M=16):
    net = ResidualDenoiser(input_dim=Z_DIM + 2, model_dim=512,
                           output_dim=Z_DIM, n_blocks=6)
    param = (OtFmEulerFmParameterizationFixed(sigma_data=1.0)
             if convention == 'fixed' else
             OtFmEulerFmParameterization(sigma_data=1.0))
    return IMM(net, param, D=Z_DIM, M=M, data_mean=mean, data_std=std,
               convention=convention).to(DEVICE)


@torch.no_grad()
def fidelity(model, Z_ref, steps, n=8192):
    g = model.sample(n, model.sampling_times(steps, device=DEVICE))
    r = Z_ref[torch.randperm(len(Z_ref), device=Z_ref.device)[:n]]
    return dict(d_mean=(g.mean(0) - r.mean(0)).abs().mean().item(),
                d_std=(g.std(0) - r.std(0)).abs().mean().item(),
                mmd=rbf_mmd(g, r))


def train(Z_tr, Z_va, a, tag):
    mean, std = Z_tr.mean(0).to(DEVICE), Z_tr.std(0).to(DEVICE)
    imm = build(a.convention, mean, std)
    ema = Ema(imm, a.ema)
    opt = torch.optim.AdamW(imm.parameters(), lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))

    Ztr, Zva = Z_tr.to(DEVICE), Z_va.to(DEVICE)
    n, bs = len(Ztr), a.bs
    total = (n // bs) * a.epochs
    warm = min(a.warmup, total // 10)

    def lr_at(step):
        if step < warm:
            return step / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    ck = f"checkpoints/{tag}.pth"
    sp = f"checkpoints/{tag}_train_state.pth"
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        imm.load_state_dict(st['model']); ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed {tag} at epoch {start} (best mmd {best:.5f})")

    t0 = time.time()
    for ep in range(start, a.epochs):
        imm.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = steps = 0
        for i in range(0, n - bs + 1, bs):
            for g in opt.param_groups:
                g['lr'] = a.lr * lr_at(gstep)
            loss = imm.compute_loss(Ztr[perm[i:i + bs]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(imm.parameters(), a.clip)
            opt.step()
            ema.update(imm)
            tot += loss.item(); steps += 1; gstep += 1

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            f = fidelity(ema.shadow, Zva, a.eval_steps)
            star = ''
            if f['mmd'] < best:
                best = f['mmd']; torch.save(ema.shadow.state_dict(), ck); star = '  *'
            print(f"  [{tag}] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:9.4f}  "
                  f"{a.eval_steps}-step  dmean {f['d_mean']:.4f}  dstd {f['d_std']:.4f}  "
                  f"mmd {f['mmd']:.5f}{star}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=imm.state_dict(), ema=ema.shadow.state_dict(),
                            opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                            best=best), sp)
    return ck, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='nvae', choices=['vae', 'nvae'])
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--convention', default='fixed', choices=['fixed', 'legacy'])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--eval_steps', type=int, default=8)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--z_fine_ch', type=int, default=None)
    a = ap.parse_args()

    Z_tr, Z_va = latent_dataset(a.model, a.dataset, a.suffix, a.z_fine_ch)
    tag = a.tag or f"{a.model}_{a.dataset}{a.suffix}_imm_{a.convention}_stage2"
    print(f"latents std {Z_tr.std():.4f} | convention={a.convention}")
    print(f"\n=== training IMM ({a.convention}) on {a.model} latents ===", flush=True)
    ck, best = train(Z_tr, Z_va, a, tag)
    print(f"best {a.eval_steps}-step latent MMD {best:.5f} -> {ck}")
