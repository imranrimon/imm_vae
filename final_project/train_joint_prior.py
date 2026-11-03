"""Prior over the JOINT latent (z_global + z_fine), which is where the
information actually lives.

Stage 2 as originally written models only z_global (64 dims). On this NVAE that
latent is already ~N(0, I) -- directly KL-regularized, and the regularization
works -- so an "expressive prior" over it has essentially nothing to learn. The
information sits in z_fine, which at sampling time is drawn from the decoder's
own conditional Gaussian p(z_fine | h_coarse), never from Stage 2.

With the compressed compressor (z_fine_ch=8) the joint latent is only
64 + 8*8*8 = 576 dims, so an MLP prior over the whole thing is finally
tractable. This script trains one and scores it against:
  * the decoder's own conditional prior (the original Stage-2 setup)
  * a temperature-swept Gaussian

Usage:
    python train_joint_prior.py --epochs 30
    python train_joint_prior.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import pandas as pd
import torch

from src.models.nvae import NVAE
from src.models.meanflow import MeanFlow, MeanFlowNet
from train_meanflow import Ema, rbf_mmd, DEVICE

SUFFIX = '_fine8'
Z_G, FINE_CH, FINE_SP = 64, 8, 8
Z_F = FINE_CH * FINE_SP * FINE_SP          # 512
D_JOINT = Z_G + Z_F                        # 576


def load_base():
    m = NVAE(z_dim=Z_G, z_fine_ch=FINE_CH).to(DEVICE)
    m.load_state_dict(torch.load(
        f"checkpoints/nvae_CelebA_kl{SUFFIX}_stage1.pth", map_location=DEVICE))
    return m.eval()


@torch.no_grad()
def joint_latents(base, split):
    cache = f"checkpoints/_joint_latents_{split}.pt"
    if os.path.exists(cache):
        return torch.load(cache)
    X = torch.load(f"data/_CelebA32_{split}.pt")
    out = []
    for i in range(0, len(X), 512):
        x = X[i:i + 512].to(DEVICE).float().div_(127.5).sub_(1.0)
        mu_g, lg_g, mu_f, lg_f, _, _ = base.encoder(x)
        z_g = mu_g + torch.randn_like(mu_g) * torch.exp(0.5 * lg_g)
        z_f = mu_f + torch.randn_like(mu_f) * torch.exp(0.5 * lg_f)
        out.append(torch.cat([z_g, z_f.flatten(1)], dim=1).cpu())
    t = torch.cat(out)
    torch.save(t, cache)
    print(f"  joint latents {split}: {tuple(t.shape)}")
    return t


def split_joint(v):
    return v[:, :Z_G], v[:, Z_G:].reshape(-1, FINE_CH, FINE_SP, FINE_SP)


@torch.no_grad()
def decode_joint(base, v):
    z_g, z_f = split_joint(v)
    out = base.decoder(z_g, z_fine=z_f)
    return out[0] if isinstance(out, tuple) else out


def train(Ztr, Zva, a, tag):
    mean, std = Ztr.mean(0).to(DEVICE), Ztr.std(0).to(DEVICE)
    net = MeanFlowNet(z_dim=D_JOINT, model_dim=a.width, n_blocks=6)
    mf = MeanFlow(net, D=D_JOINT, variant='base',
                  data_mean=mean, data_std=std).to(DEVICE)
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

    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_train_state.pth"
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
            for g in opt.param_groups:
                g['lr'] = a.lr * lr_at(gstep)
            loss = mf.compute_loss(Ztr[perm[i:i + bs]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mf.parameters(), 1.0)
            opt.step(); ema.update(mf)
            tot += loss.item(); steps += 1; gstep += 1

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                g = ema.shadow.sample(4096, steps=1)
            r = Zva[torch.randperm(len(Zva), device=DEVICE)[:4096]]
            m = rbf_mmd(g, r)
            ds = (g.std(0) - r.std(0)).abs().mean().item()
            star = ''
            if m < best:
                best = m; torch.save(ema.shadow.state_dict(), ck); star = '  *'
            print(f"  [{tag}] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:9.3f}  "
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
    print(f"joint latents: {tuple(Ztr.shape)}  std {Ztr.std():.4f}")
    print(f"  z_global std {Ztr[:, :Z_G].std():.4f} | z_fine std {Ztr[:, Z_G:].std():.4f}")

    tag = 'nvae_CelebA_fine8_joint_meanflow'
    if not a.eval_only:
        print(f"\n=== training JOINT MeanFlow prior (D={D_JOINT}) ===", flush=True)
        train(Ztr, Zva, a, tag)

    # ---------------- evaluation ----------------
    from ab_sampler_eval import Fid, gen_images
    fid = Fid()
    rows = []

    def score(name, nfe, sampler, joint):
        torch.manual_seed(0)
        if joint:
            imgs = []
            done = 0
            while done < a.fake:
                b = min(500, a.fake - done)
                with torch.no_grad():
                    imgs.append(decode_joint(base, sampler(b)).cpu())
                done += b
            imgs = torch.cat(imgs)[:a.fake]
        else:
            imgs = gen_images(a.fake, 'nvae', base, sampler)
        f = fid.score(imgs, a.real)
        rows.append(dict(Prior=name, NFE=nfe, FID=round(f, 3)))
        print(f"  {name:34s} NFE={nfe:>3}  FID={f:7.3f}", flush=True)

    # baseline: z_global ~ N(0,sigma I), z_fine from the decoder's own prior
    for sg in [1.0, 1.3, 1.5]:
        score(f'z_global N(0,{sg}) + decoder p(z_fine)', 0,
              lambda b, s=sg: s * torch.randn(b, Z_G, device=DEVICE), joint=False)

    ckp = f"checkpoints/{tag}.pth"
    if os.path.exists(ckp):
        net = MeanFlowNet(z_dim=D_JOINT, model_dim=a.width, n_blocks=6)
        mf = MeanFlow(net, D=D_JOINT, variant='base').to(DEVICE)
        mf.load_state_dict(torch.load(ckp, map_location=DEVICE)); mf.eval()
        for st in [1, 2, 4]:
            score('JOINT MeanFlow (z_global + z_fine)', st,
                  lambda b, st=st: mf.sample(b, steps=st), joint=True)

    df = pd.DataFrame(rows)
    out = 'results/CelebA/joint_prior.csv'
    df.to_csv(out, index=False)
    print("\n=== JOINT PRIOR COMPARISON (compressed NVAE, z_fine=512) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")
