"""Stage-2 prior on the proper NVAE's top latent group.

The proper NVAE already has a learned, jointly-trained conditional prior for
every group after the first, yet ancestral sampling reaches only FID 74.5
against a ceiling of 2.01 -- a 37x gap, the largest in this study. This asks
whether that gap is recoverable by a Stage-2 prior over the ONE group that is
genuinely unconditional (group 1, 8x8x8 = 512 dims at 32x32).

If it is, joint training of a conditional hierarchy is not sufficient to close
the prior--posterior gap and our diagnostic applies to NVAE-style models too.
If it is not, the gap lives in the conditional groups and a top-group prior
cannot reach it -- which would be an equally informative negative.

Usage:
    python train_nvae_prior.py --epochs 60
    python train_nvae_prior.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch

from src.models.nvae_proper import ProperNVAE
from src.models.conv_meanflow import ConvMeanFlowNet
from src.models.imm_core import IMM, OtFmEulerFmParameterizationFixed
from train_stage1_fast import image_cache, to_float
from train_meanflow import Ema, rbf_mmd
from fid_standard import StandardFid, run

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvIMMNet(torch.nn.Module):
    """IMM hands its network one [B, D+2] tensor; unpack and run the conv trunk."""

    def __init__(self, C, S, D, width=192, blocks=6):
        super().__init__()
        self.D = D
        self.net = ConvMeanFlowNet(channels=C, spatial=S, width=width,
                                   n_blocks=blocks, cond_dim=0, time_scale=1.0)

    def forward(self, inp):
        z, s, t = inp[:, :self.D], inp[:, self.D:self.D + 1], inp[:, self.D + 1:]
        return self.net(z, s / 1000.0, t / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--z_ch', type=int, default=8)
    ap.add_argument('--scales', type=int, default=3)
    ap.add_argument('--gps', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    a = ap.parse_args()

    m = ProperNVAE(base=a.base, z_ch=a.z_ch, scales=a.scales,
                   groups_per_scale=a.gps, img=a.image_size).to(DEVICE)
    m.load_state_dict(torch.load(
        f"checkpoints/nvaeP_{a.dataset}_s{a.scales}g{a.gps}z{a.z_ch}.pth",
        map_location=DEVICE))
    m.eval()
    C, S, _ = m.top_shape()
    D = C * S * S
    print(f"NVAE top group: {C}x{S}x{S} = {D} dims")

    # ---- extract the top-group posterior once -------------------------------
    cache = f"checkpoints/_nvaeP_top_{a.dataset}.pt"
    if os.path.exists(cache):
        Z = torch.load(cache)
    else:
        X = image_cache(a.dataset, 'train', a.image_size)
        outs = []
        for i in range(0, len(X), 512):
            outs.append(m.encode_top(to_float(X[i:i + 512]).to(DEVICE))
                        .flatten(1).cpu())
        Z = torch.cat(outs)
        torch.save(Z, cache)
    print(f"top-group latents {tuple(Z.shape)}  std {Z.std():.4f}")
    zn = (Z - Z.mean(0)) / Z.std(0)
    print(f"  excess kurtosis {(((zn - zn.mean(0)) / zn.std(0)) ** 4).mean() - 3:+.4f}")

    Ztr, Zva = Z[:-10000].to(DEVICE), Z[-10000:].to(DEVICE)
    mean, sd = Ztr.mean(0), Ztr.std(0)

    net = ConvIMMNet(C, S, D)
    prior = IMM(net, OtFmEulerFmParameterizationFixed(1.0), D=D, M=16,
                data_mean=mean, data_std=sd, convention='fixed').to(DEVICE)
    ema = Ema(prior, 0.999)
    opt = torch.optim.AdamW(prior.parameters(), lr=a.lr, betas=(0.9, 0.99))
    tag = f"nvaeP_{a.dataset}_topprior"
    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_state.pth"
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        prior.load_state_dict(st['m']); ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed at epoch {start} (best mmd {best:.5f})")

    n, bs = len(Ztr), a.bs
    total = (n // bs) * a.epochs
    warm = min(500, total // 10)

    def lr_at(g):
        if g < warm:
            return g / max(warm, 1)
        p = (g - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    if not a.eval_only:
        t0 = time.time()
        for ep in range(start, a.epochs):
            prior.train()
            perm = torch.randperm(n, device=DEVICE)
            tot = steps = 0
            for i in range(0, n - bs + 1, bs):
                for g in opt.param_groups:
                    g['lr'] = a.lr * lr_at(gstep)
                loss = prior.compute_loss(Ztr[perm[i:i + bs]])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                opt.step(); ema.update(prior)
                tot += loss.item(); steps += 1; gstep += 1

            if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
                with torch.no_grad():
                    g = ema.shadow.sample(4096, ema.shadow.sampling_times(4, device=DEVICE))
                r = Zva[torch.randperm(len(Zva), device=DEVICE)[:4096]]
                mmd = rbf_mmd(g, r)
                star = ''
                if mmd < best:
                    best = mmd; torch.save(ema.shadow.state_dict(), ck); star = '  *'
                print(f"  ep {ep+1:3d}/{a.epochs}  loss {tot/steps:.4f}  "
                      f"dstd {(g.std(0)-r.std(0)).abs().mean():.4f}  "
                      f"mmd {mmd:.5f}{star}  ({time.time()-t0:.0f}s)", flush=True)
                torch.save(dict(m=prior.state_dict(), ema=ema.shadow.state_dict(),
                                opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                                best=best), sp)

    prior.load_state_dict(torch.load(ck, map_location=DEVICE)); prior.eval()

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    cfg = []
    for t in [0.8, 1.0]:
        cfg.append(('nvae_proper', f'ancestral t={t} [baseline]', 0,
                    lambda b, tt=t: m.sample(b, DEVICE, t=tt)))
    for nfe in [1, 4, 8]:
        def mk(b, k=nfe):
            z1 = prior.sample(b, prior.sampling_times(k, device=DEVICE))
            return m.sample_given_top(z1.view(b, C, S, S), t=1.0)
        cfg.append(('nvae_proper', 'IMM top-group prior', nfe, mk))
    df = run(cfg, 50000, fid, f'results/{a.dataset}/nvae_proper.csv',
             variant='topprior')
    print("\n=== NVAE + STAGE-2 TOP-GROUP PRIOR (ceiling 2.01) ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
