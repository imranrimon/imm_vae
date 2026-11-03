"""Fast Stage-1 trainer: caches decoded images once instead of re-decoding JPEGs.

src/training/stage1.py re-decodes all 162k CelebA JPEGs every epoch, which
dominates runtime (~9 min/epoch). At 32x32 the whole training split is ~500 MB
of uint8, so it fits in RAM (or VRAM) and epochs become seconds.

Also exposes the knobs that matter for actually compressing z_fine:
  --z_fine_ch   fine-latent channels (8 -> 8x8x8 = 512 dims, vs the default 8192)
  --free_bits   per-dim KL floor, so raising the KL weight regularizes
                instead of collapsing the code
  --kl_warmup   linear ramp on the KL weight

Usage:
    python train_stage1_fast.py --z_fine_ch 8 --kl_weight 0.01 --free_bits 0.05
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch
import torch.nn.functional as F

from src.data import get_dataloader
from src.models.nvae import NVAE
from src.models.vae import VAE
from src.training.stage1 import (kl_std_normal_perdim, kl_general_perdim,
                                 apply_free_bits, loss_vae_mmd)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_cache(dataset, split, image_size=32):
    """Decode once to a uint8 tensor [N,3,S,S]; reused by every later run."""
    path = f"data/_{dataset}{image_size}_{split}.pt"
    if os.path.exists(path):
        return torch.load(path)
    print(f"building image cache for {dataset}/{split} (one-time)...", flush=True)
    loader = get_dataloader(dataset, batch_size=512, split=split, drop_last=False,
                            image_size=image_size)
    chunks = []
    for x, _ in loader:
        # stored as uint8 in [0,255]; converted back to [-1,1] on use
        chunks.append(((x * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8))
    t = torch.cat(chunks)
    torch.save(t, path)
    print(f"  cached {tuple(t.shape)} -> {path}")
    return t


def to_float(u8):
    return u8.float().div_(127.5).sub_(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='nvae', choices=['nvae', 'vae'])
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--method', default='kl', choices=['kl', 'mmd'])
    ap.add_argument('--z_dim', type=int, default=64)
    ap.add_argument('--z_fine_ch', type=int, default=None)
    ap.add_argument('--kl_weight', type=float, default=0.00025)
    ap.add_argument('--free_bits', type=float, default=0.0)
    ap.add_argument('--kl_warmup', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    Xtr = image_cache(a.dataset, 'train').to(DEVICE)
    Xva = image_cache(a.dataset, 'test').to(DEVICE)
    print(f"images: train={tuple(Xtr.shape)} val={tuple(Xva.shape)}")

    if a.model == 'nvae':
        model = NVAE(z_dim=a.z_dim, z_fine_ch=a.z_fine_ch).to(DEVICE)
        g, f = model.latent_dims(32)
        print(f"NVAE latents: z_global={g}  z_fine={f}  total={g+f}  "
              f"({(g+f)/3072:.3f}x the 32x32x3 image)")
    else:
        model = VAE(latent_dim=a.z_dim).to(DEVICE)

    suffix = f"_fine{a.z_fine_ch}" if (a.model == 'nvae' and a.z_fine_ch) else ''
    tag = a.tag or f"{a.model}_{a.dataset}_{a.method}{suffix}_stage1"
    ck = f"checkpoints/{tag}.pth"
    sp = f"checkpoints/{tag}_train_state.pth"

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n, bs = len(Xtr), a.bs
    total = (n // bs) * a.epochs
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed at epoch {start} (best val {best:.4f})")

    def step_loss(x, kl_w):
        if a.model == 'vae':
            rec, mu, lv = model(x)
            r = F.mse_loss(rec, x, reduction='sum') / x.shape[0]
            if a.method == 'kl':
                return r + kl_w * apply_free_bits(
                    kl_std_normal_perdim(mu, lv), a.free_bits), r
            z = model.reparameterize(mu, lv)
            return r + 10.0 * loss_vae_mmd(z, torch.randn_like(z)), r
        out = model(x)
        r = F.mse_loss(out['x_rec'], x, reduction='sum') / x.shape[0]
        kl_f = apply_free_bits(
            kl_general_perdim(out['mu_f'], out['log_f'],
                              out['mu_prior_f'], out['log_prior_f']).flatten(),
            a.free_bits)
        if a.method == 'kl':
            kl_g = apply_free_bits(
                kl_std_normal_perdim(out['mu_g'], out['log_g']), a.free_bits)
            return r + kl_w * (kl_g + kl_f), r
        z_g = out['z_g']
        mmd = loss_vae_mmd(z_g, torch.randn_like(z_g))
        return r + 10.0 * mmd + kl_w * kl_f, r

    t0 = time.time()
    for ep in range(start, a.epochs):
        kl_w = a.kl_weight * (min(1.0, (ep + 1) / a.kl_warmup) if a.kl_warmup else 1.0)
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tl = tr = steps = 0
        for i in range(0, n - bs + 1, bs):
            x = to_float(Xtr[perm[i:i + bs]])
            loss, rec = step_loss(x, kl_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item(); tr += rec.item(); steps += 1; gstep += 1

        model.eval()
        with torch.no_grad():
            vl = vr = vs = 0
            for i in range(0, len(Xva) - bs + 1, bs):
                x = to_float(Xva[i:i + bs])
                l, r = step_loss(x, kl_w)
                vl += l.item(); vr += r.item(); vs += 1
            vl /= max(vs, 1); vr /= max(vs, 1)
            # per-pixel MSE on the [0,255] scale, comparable to the paper's Table 1
            mse255 = vr / 3072 * (127.5 ** 2)

        star = ''
        if vl < best:
            best = vl; torch.save(model.state_dict(), ck); star = '  *'
        print(f"  ep {ep+1:3d}/{a.epochs}  kl_w {kl_w:.5f}  train {tl/steps:8.3f}  "
              f"val {vl:8.3f}  recon {vr:7.3f}  MSE255 {mse255:7.3f}{star}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        epoch=ep + 1, gstep=gstep, best=best), sp)
    print(f"best val {best:.4f} -> {ck}")


if __name__ == "__main__":
    main()
