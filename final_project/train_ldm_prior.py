"""Priors over the LDM-autoencoder latent, with CONV backbones for both methods.

This is the fair test that the flat-MLP experiments were not. Earlier priors were
MLPs over a flattened vector, and both MeanFlow and IMM degraded badly with
dimension (D=2 excellent, D=32 good, D=512 failed). Published MeanFlow/IMM
results use conv/U-Net backbones on spatially structured latents, so this runs
both methods with the conv backbone on the 8x8x4 LDM-AE latent.

Context (standard FID, 50k vs 50k):
    LDM-AE ceiling (decode true latents)        1.78
    LDM-AE per-dim Gaussian prior             186.75
So there is ~185 FID of headroom -- unlike the NVAE's z_global, where the
aggregate posterior was already ~N(0, I) and there was nothing to learn.

Usage:
    python train_ldm_prior.py --method meanflow --epochs 60
    python train_ldm_prior.py --method imm --epochs 60
    python train_ldm_prior.py --method meanflow --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch
import torch.nn as nn

from src.models.conv_meanflow import ConvMeanFlowNet
from src.models.meanflow import MeanFlow
from src.models.imm_core import IMM, OtFmEulerFmParameterizationFixed
from src.models.ldm_ae import LdmAutoencoder
from train_meanflow import Ema, rbf_mmd, DEVICE

# resolution-dependent; set from the latent tensor at load time
C, S, D = 4, 8, 256


class ConvIMMNet(nn.Module):
    """Adapter: IMM hands its network a single [B, D+2] tensor.

    Splits it back into (z, c_noise(s), c_noise(t)), undoes IMM's 1000x time
    scaling, and runs the same conv trunk MeanFlow uses so the two methods are
    compared on identical capacity.
    """

    def __init__(self, width=192, blocks=6):
        super().__init__()
        self.net = ConvMeanFlowNet(channels=C, spatial=S, width=width,
                                   n_blocks=blocks, cond_dim=0, time_scale=1.0)

    def forward(self, inp):
        z, s, t = inp[:, :D], inp[:, D:D + 1], inp[:, D + 1:D + 2]
        return self.net(z, s / 1000.0, t / 1000.0)


def load_latents(dataset='CelebA', image_size=32, sigma=False):
    global C, S, D
    suffix = '' if image_size == 32 else str(image_size)
    sv = '_sigma' if sigma else ''
    d = torch.load(f'checkpoints/_ldmae_latents_{dataset}{suffix}_4{sv}.pt')
    z = d['z']                      # [N, C, S, S]
    C, S = z.shape[1], z.shape[2]
    D = C * S * S
    z = z.reshape(len(z), -1)
    return z, float(d['scale'])


def build(method, mean, std, width, blocks, M=16):
    if method == 'meanflow':
        net = ConvMeanFlowNet(channels=C, spatial=S, width=width,
                              n_blocks=blocks, cond_dim=0, time_scale=1.0)  # noqa: F824
        return MeanFlow(net, D=D, variant='base', data_mean=mean, data_std=std,
                        target_clip=60.0).to(DEVICE)
    net = ConvIMMNet(width=width, blocks=blocks)
    return IMM(net, OtFmEulerFmParameterizationFixed(1.0), D=D, M=M,
               data_mean=mean, data_std=std, convention='fixed').to(DEVICE)


@torch.no_grad()
def gen(model, method, n, steps):
    if method == 'meanflow':
        return model.sample(n, steps=steps)
    return model.sample(n, model.sampling_times(steps, device=DEVICE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='meanflow', choices=['meanflow', 'imm'])
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--sigma_vae', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--width', type=int, default=192)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    ap.add_argument('--fake', type=int, default=50000)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--M', type=int, default=16,
                    help='IMM particles per group; groups G = batch/M')
    a = ap.parse_args()

    Z, scale = load_latents(a.dataset, a.image_size, a.sigma_vae)
    n_val = 10000
    Ztr, Zva = Z[:-n_val].to(DEVICE), Z[-n_val:].to(DEVICE)
    mean, std = Ztr.mean(0), Ztr.std(0)
    print(f"LDM-AE latents {tuple(Z.shape)}  std {Z.std():.4f}  (scale {scale:.4f})")
    zn = (Ztr - mean) / std
    print(f"normalized: excess kurtosis {(((zn - zn.mean(0)) / zn.std(0)) ** 4).mean() - 3:+.4f} "
          f"(0 = Gaussian; this is what a prior must capture)")

    model = build(a.method, mean, std, a.width, a.blocks, a.M)
    print(f"{a.method} conv prior: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    ema = Ema(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99))

    # Must carry dataset AND resolution AND capacity: without them every run
    # writes to the same file and silently overwrites the previous one.
    res = '' if a.image_size == 32 else str(a.image_size)
    res += '_sigma' if a.sigma_vae else ''
    cap = '' if (a.width, a.blocks) == (192, 6) else f"_w{a.width}b{a.blocks}"
    tag = a.tag or f"ldmae_{a.dataset}{res}_z4_{a.method}_conv{cap}"
    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_train_state.pth"
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        model.load_state_dict(st['model']); ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed at epoch {start} (best mmd {best:.5f})")

    nn_, bs = len(Ztr), a.bs
    total = (nn_ // bs) * a.epochs
    warm = min(500, total // 10)

    def lr_at(g):
        if g < warm:
            return g / max(warm, 1)
        p = (g - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    if not a.eval_only:
        print(f"\n=== {a.method} (conv) on the LDM-AE latent ===", flush=True)
        t0 = time.time()
        for ep in range(start, a.epochs):
            model.train()
            perm = torch.randperm(nn_, device=DEVICE)
            tot = steps = 0
            for i in range(0, nn_ - bs + 1, bs):
                for g in opt.param_groups:
                    g['lr'] = a.lr * lr_at(gstep)
                loss = model.compute_loss(Ztr[perm[i:i + bs]])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ema.update(model)
                tot += loss.item(); steps += 1; gstep += 1

            if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
                g = gen(ema.shadow, a.method, 4096, 1 if a.method == 'meanflow' else 8)
                r = Zva[torch.randperm(len(Zva), device=DEVICE)[:4096]]
                m = rbf_mmd(g, r)
                ds = (g.std(0) - r.std(0)).abs().mean().item()
                star = ''
                if m < best:
                    best = m; torch.save(ema.shadow.state_dict(), ck); star = '  *'
                print(f"  [{a.method}] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:9.4f}  "
                      f"dstd {ds:.4f}  mmd {m:.5f}{star}  ({time.time()-t0:.0f}s)", flush=True)
                torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(),
                                opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                                best=best), sp)

    model.load_state_dict(torch.load(ck, map_location=DEVICE)); model.eval()

    ae = LdmAutoencoder(base=128, z_ch=C, n_res=2).to(DEVICE)
    ae_ck = (f'checkpoints/ldmae_{a.dataset}_z4.pth' if a.image_size == 32
             else f'checkpoints/ldmae_{a.dataset}{a.image_size}_z4.pth')
    if a.sigma_vae:
        ae_ck = f'checkpoints/ldmae_{a.dataset}_z4_sigma.pth'
    ae.load_state_dict(torch.load(ae_ck, map_location=DEVICE))
    ae.eval()

    from fid_standard import StandardFid, run
    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    cfg = []
    for st in ([1, 2, 4, 8] if a.method == 'meanflow' else [1, 2, 4, 8, 16]):
        def mk(b, st=st):
            z = gen(model, a.method, b, st)
            return ae.decode(z.reshape(-1, C, S, S))
        cfg.append((f'ldm_ae_{a.dataset}', f'{a.method} conv prior', st, mk))
    print()
    df = run(cfg, a.fake, fid,
             f"results/{a.dataset}/ldm_prior{'' if a.image_size == 32 else a.image_size}.csv",
             variant=f"w{a.width}b{a.blocks}{'_sigma' if a.sigma_vae else ''}")
    print(f"\n=== CONV PRIOR ON LDM-AE LATENT (ceiling 1.78 | Gaussian 186.75) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
