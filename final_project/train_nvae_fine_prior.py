"""Stage-2 priors on the NVAE's FINE latent groups, conditioned on the top-down state.

The mixed-sampling sweep localised the gap precisely: with 6 groups, sampling
groups 1-3 from the prior costs nothing (FID 3.11 -> 2.56), and groups 4-6 cost
everything (29.65 -> 90.17 -> 144.35). A Stage-2 prior on group 1 therefore did
nothing, because group 1 was already free. This puts the prior where the loss is.

Two design points follow from the hierarchy:

  Conditional, not marginal.  p(z_l | z_<l) already exists and is jointly
  trained; a replacement must condition on the same top-down state h_l or it
  discards information NVAE already has.

  Residual, not absolute.  We model eps_l = (z_l - mu_p) / sigma_p rather than
  z_l. If the prior learns eps ~ N(0, I) it reproduces NVAE's own conditional
  Gaussian exactly, so the baseline is a floor rather than a risk -- the same
  parameterisation that was necessary to get anywhere on the z_fine residual.

The top-down states are 64-channel feature maps up to 32x32, far too large to
cache for 162k images, so each batch runs the NVAE forward to produce its own
(h_l, eps_l) pairs.

Usage:
    python train_nvae_fine_prior.py --norm group --epochs 30
    python train_nvae_fine_prior.py --norm group --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch
import torch.nn as nn

from src.models.nvae_proper import ProperNVAE
from src.models.conv_meanflow import SinusoidalPosEmb, ConvResBlock
from src.models.meanflow import MeanFlow
from train_stage1_fast import image_cache, to_float
from train_meanflow import Ema
from fid_standard import StandardFid, run

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINE = (3, 4, 5)          # 0-indexed groups 4,5,6 -- the ones that cost


class SpatialCondNet(nn.Module):
    """u(eps, r, t | h): conv trunk over the latent grid with the top-down state
    concatenated as extra channels."""

    def __init__(self, z_ch=8, cond_ch=64, width=128, blocks=4, time_dim=64):
        super().__init__()
        self.z_ch = z_ch
        self.time_emb = SinusoidalPosEmb(time_dim)
        emb = width * 2
        self.emb_mlp = nn.Sequential(nn.Linear(2 * time_dim, emb), nn.SiLU(),
                                     nn.Linear(emb, emb))
        self.conv_in = nn.Conv2d(z_ch + cond_ch, width, 3, padding=1)
        self.blocks = nn.ModuleList([ConvResBlock(width, emb) for _ in range(blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv2d(width, z_ch, 3, padding=1))
        self.cond = None                      # set by the caller each step

    def forward(self, z, r, t, cond=None):
        h = self.cond
        B = z.shape[0]
        S = h.shape[-1]
        x = z.reshape(B, self.z_ch, S, S)
        e = self.emb_mlp(torch.cat([self.time_emb(r.reshape(-1)),
                                    self.time_emb(t.reshape(-1))], -1))
        y = self.conv_in(torch.cat([x, h], 1))
        for b in self.blocks:
            y = b(y, e)
        return self.out(y).reshape(B, -1)


def build_priors(m, width, blocks):
    priors, r = {}, m.img // 2 ** (m.scales - 1)
    res = []
    for s in range(m.scales):
        for _ in range(m.gps):
            res.append(r)
        r *= 2
    for gi in FINE:
        S = res[gi]
        D = m.z_ch * S * S
        net = SpatialCondNet(z_ch=m.z_ch, cond_ch=m.h0.shape[1],
                             width=width, blocks=blocks)
        p = MeanFlow(net, D=D, variant='base',
                     data_mean=torch.zeros(D), data_std=torch.ones(D),
                     target_clip=4.0 * D ** 0.5).to(DEVICE)
        priors[gi] = p
    return priors, res


def fine_pairs(m, x):
    """One NVAE forward, yielding (group, h_l, eps_l) for each fine group."""
    feats = m._encode(x)
    B = x.shape[0]
    h = m.h0.expand(B, -1, -1, -1)
    out, gi = [], 0
    for s in range(m.scales):
        ef = feats[s]
        for _ in range(m.gps):
            mu_p, lv_p = m.prior[gi](h).chunk(2, 1)
            lv_p = lv_p.clamp(-8, 8)
            dmu, dlv = m.post[gi](torch.cat([h, ef], 1)).chunk(2, 1)
            dlv = dlv.clamp(-8, 8)
            mu_q = mu_p + dmu * torch.exp(lv_p)
            lv_q = lv_p + dlv
            z = mu_q + torch.randn_like(mu_q) * torch.exp(lv_q)
            if gi in FINE:
                eps = (z - mu_p) / torch.exp(lv_p)
                out.append((gi, h.detach(), eps.detach()))
            h = m.cells[gi](m.merge[gi](torch.cat([h, z], 1)))
            gi += 1
        if s < m.scales - 1:
            h = m.up[s](h)
    return out


@torch.no_grad()
def sample_with_fine_priors(m, priors, n, nfe=4, t=1.0):
    h = m.h0.expand(n, -1, -1, -1)
    gi = 0
    for s in range(m.scales):
        for _ in range(m.gps):
            mu_p, lv_p = m.prior[gi](h).chunk(2, 1)
            lv_p = lv_p.clamp(-8, 8)
            if gi in priors:
                p = priors[gi]
                p.network.cond = h
                eps = p.sample(n, steps=nfe).view_as(mu_p)
                z = mu_p + eps * torch.exp(lv_p)
            else:
                z = mu_p + t * torch.randn_like(mu_p) * torch.exp(lv_p)
            h = m.cells[gi](m.merge[gi](torch.cat([h, z], 1)))
            gi += 1
        if s < m.scales - 1:
            h = m.up[s](h)
    return torch.tanh(m.out(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--z_ch', type=int, default=8)
    ap.add_argument('--scales', type=int, default=3)
    ap.add_argument('--gps', type=int, default=2)
    ap.add_argument('--norm', default='group', choices=['batch', 'group'])
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--max_train', type=int, default=0,
                    help='cap the training set; each image yields 3 (h,eps) pairs')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    a = ap.parse_args()

    nm = '' if a.norm == 'batch' else '_gn'
    m = ProperNVAE(base=a.base, z_ch=a.z_ch, scales=a.scales,
                   groups_per_scale=a.gps, img=a.image_size, norm=a.norm).to(DEVICE)
    m.load_state_dict(torch.load(
        f"checkpoints/nvaeP_{a.dataset}_s{a.scales}g{a.gps}z{a.z_ch}{nm}.pth",
        map_location=DEVICE))
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)

    priors, res = build_priors(m, a.width, a.blocks)
    print(f"fine-group priors on groups {[g+1 for g in FINE]} "
          f"(resolutions {[res[g] for g in FINE]}), "
          f"{sum(sum(q.numel() for q in p.parameters()) for p in priors.values())/1e6:.2f}M params")

    tag = f"nvaeP_{a.dataset}{nm}_fineprior"
    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_state.pth"
    params = [q for p in priors.values() for q in p.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99))
    emas = {g: Ema(p, 0.999) for g, p in priors.items()}
    start, gstep = 0, 0
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        for g in priors:
            priors[g].load_state_dict(st['m'][str(g)])
            emas[g].shadow.load_state_dict(st['ema'][str(g)])
        opt.load_state_dict(st['opt']); start, gstep = st['epoch'], st['gstep']
        print(f"  resumed at epoch {start}")

    X = image_cache(a.dataset, 'train', a.image_size)
    if a.max_train:
        X = X[:a.max_train]
    n, bs = len(X), a.bs
    print(f"training on {n} images ({n * len(FINE)} conditional pairs per epoch)")
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
            perm = torch.randperm(n)
            tot = steps = 0
            for i in range(0, n - bs + 1, bs):
                x = to_float(X[perm[i:i + bs]]).to(DEVICE)
                with torch.no_grad():
                    pairs = fine_pairs(m, x)
                for g in opt.param_groups:
                    g['lr'] = a.lr * lr_at(gstep)
                loss = 0
                for gi, hcond, eps in pairs:
                    p = priors[gi]
                    p.network.cond = hcond
                    loss = loss + p.compute_loss(eps.flatten(1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                for g_, e_ in emas.items():
                    e_.update(priors[g_])
                tot += float(loss); steps += 1; gstep += 1

            if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
                print(f"  ep {ep+1:3d}/{a.epochs}  loss {tot/steps:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                torch.save(dict(m={str(g): p.state_dict() for g, p in priors.items()},
                                ema={str(g): e.shadow.state_dict() for g, e in emas.items()},
                                opt=opt.state_dict(), epoch=ep + 1, gstep=gstep), sp)
                torch.save({str(g): e.shadow.state_dict() for g, e in emas.items()}, ck)

    sd = torch.load(ck, map_location=DEVICE)
    for g in priors:
        priors[g].load_state_dict(sd[str(g)]); priors[g].eval()

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    cfg = [('nvae_proper', f'ancestral t=1.0 [{a.norm} baseline]', 0,
            lambda b: m.sample(b, DEVICE, t=1.0)),
           ('nvae_proper', f'ancestral t=0.8 [{a.norm} baseline]', 0,
            lambda b: m.sample(b, DEVICE, t=0.8))]
    for nfe in [1, 4]:
        cfg.append(('nvae_proper', 'fine-group IMM priors (4,5,6)', nfe,
                    lambda b, k=nfe: sample_with_fine_priors(m, priors, b, nfe=k)))
    df = run(cfg, 50000, fid, f'results/{a.dataset}/nvae_proper.csv',
             variant=f'fineprior{nm}')
    print(f"\n=== NVAE ({a.norm}) + FINE-GROUP PRIORS (ceiling "
          f"{1.67 if a.norm == 'group' else 2.01}) ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
