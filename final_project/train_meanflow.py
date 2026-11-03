"""Train MeanFlow / improved-MeanFlow latent priors, with EMA + resume.

Stage 1 is frozen, so the latent dataset is extracted ONCE and cached rather
than re-encoding every image every epoch (which is what src/training/stage2.py
does). On CelebA that is ~41 MB of float32 and makes training minutes, not hours.

Model selection uses latent fidelity (per-dim moment error + kernel MMD against
held-out encoder latents), which is cheap and, unlike FID, actually measures
what a prior is supposed to do.

Usage:
    python train_meanflow.py --model nvae --variant meanflow --epochs 120
    python train_meanflow.py --model nvae --variant meanflow --epochs 120 --resume
    python train_meanflow.py --model nvae --variant meanflow_improved --sweep
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import copy
import json
import math
import time

import torch

from src.data import get_dataloader
from src.models.vae import VAE
from src.models.nvae import NVAE
from src.models.meanflow import MeanFlow, MeanFlowNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_DIM = 64


# ------------------------------------------------------------------ latents
def load_base(model_type, dataset, method='kl', suffix='', z_fine_ch=None):
    m = (VAE(latent_dim=Z_DIM) if model_type == 'vae'
         else NVAE(z_dim=Z_DIM, z_fine_ch=z_fine_ch))
    m = m.to(DEVICE)
    m.load_state_dict(torch.load(
        f"checkpoints/{model_type}_{dataset}_{method}{suffix}_stage1.pth",
        map_location=DEVICE))
    return m.eval()


@torch.no_grad()
def encode_split(base, model_type, dataset, split):
    # Prefer the decoded uint8 cache built by train_stage1_fast.py: encoding
    # from it takes seconds instead of re-decoding 162k JPEGs.
    cpath = f"data/_{dataset}32_{split}.pt"
    if os.path.exists(cpath):
        X = torch.load(cpath)
        zs = []
        for i in range(0, len(X), 512):
            x = X[i:i + 512].to(DEVICE).float().div_(127.5).sub_(1.0)
            if model_type == 'vae':
                mu, logvar = base.encode(x)
            else:
                mu, logvar, _, _, _, _ = base.encoder(x)
            zs.append((mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)).cpu())
        return torch.cat(zs)

    loader = get_dataloader(dataset, batch_size=256, split=split, drop_last=False)
    zs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        if model_type == 'vae':
            mu, logvar = base.encode(x)
        else:
            mu, logvar, _, _, _, _ = base.encoder(x)
        zs.append((mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)).cpu())
    return torch.cat(zs)


def latent_dataset(model_type, dataset, suffix='', z_fine_ch=None):
    cache = f"checkpoints/_latents_{model_type}_{dataset}{suffix}.pt"
    if os.path.exists(cache):
        d = torch.load(cache)
        return d['train'], d['val']
    base = load_base(model_type, dataset, suffix=suffix, z_fine_ch=z_fine_ch)
    print("extracting latents (one pass over the frozen compressor)...")
    tr = encode_split(base, model_type, dataset, 'train')
    va = encode_split(base, model_type, dataset, 'test')
    torch.save({'train': tr, 'val': va}, cache)
    return tr, va


# ------------------------------------------------------------------ fidelity
def rbf_mmd(x, y, n=4096):
    """Unbiased-ish RBF MMD^2 with median-heuristic bandwidth."""
    x = x[torch.randperm(len(x), device=x.device)[:n]]
    y = y[torch.randperm(len(y), device=y.device)[:n]]
    z = torch.cat([x, y])
    d2 = torch.cdist(z, z).pow(2)
    med = d2[d2 > 0].median().clamp(min=1e-8)
    k = torch.exp(-d2 / med)
    m = len(x)
    return (k[:m, :m].mean() + k[m:, m:].mean() - 2 * k[:m, m:].mean()).item()


@torch.no_grad()
def fidelity(model, Z_ref, steps=1, n=8192):
    g = model.sample(n, steps=steps)
    r = Z_ref[torch.randperm(len(Z_ref), device=Z_ref.device)[:n]]
    return dict(
        d_mean=(g.mean(0) - r.mean(0)).abs().mean().item(),
        d_std=(g.std(0) - r.std(0)).abs().mean().item(),
        mmd=rbf_mmd(g, r),
    )


# ------------------------------------------------------------------ EMA
class Ema:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p, alpha=1 - self.decay)
            else:
                s.copy_(p)


# ------------------------------------------------------------------ training
def build(variant, mean, std, **kw):
    net = MeanFlowNet(z_dim=Z_DIM, model_dim=512, n_blocks=6)
    return MeanFlow(net, D=Z_DIM,
                    variant='base' if variant == 'meanflow' else 'improved',
                    data_mean=mean, data_std=std, **kw).to(DEVICE)


def train(variant, Z_tr, Z_va, args, tag, mf_kw=None, quiet=False):
    mean, std = Z_tr.mean(0).to(DEVICE), Z_tr.std(0).to(DEVICE)
    mf = build(variant, mean, std, **(mf_kw or {}))
    ema = Ema(mf, args.ema)
    opt = torch.optim.AdamW(mf.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))

    Ztr, Zva = Z_tr.to(DEVICE), Z_va.to(DEVICE)
    n, bs = len(Ztr), args.bs
    spe = n // bs
    total = spe * args.epochs
    warm = min(args.warmup, total // 10)

    def lr_at(step):
        if step < warm:
            return step / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    ck = f"checkpoints/{tag}.pth"
    state_path = f"checkpoints/{tag}_train_state.pth"
    start_ep, gstep, best = 0, 0, float('inf')
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=DEVICE)
        mf.load_state_dict(st['model'])
        ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start_ep, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed {tag} at epoch {start_ep} (best mmd {best:.5f})")

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        mf.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = steps = 0
        for i in range(0, n - bs + 1, bs):
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_at(gstep)
            loss = mf.compute_loss(Ztr[perm[i:i + bs]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mf.parameters(), args.clip)
            opt.step()
            ema.update(mf)
            tot += loss.item()
            steps += 1
            gstep += 1

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            f = fidelity(ema.shadow, Zva, steps=1)
            score = f['mmd']
            star = ''
            if score < best:
                best = score
                torch.save(ema.shadow.state_dict(), ck)
                star = '  *'
            if not quiet:
                print(f"  [{tag}] ep {ep+1:3d}/{args.epochs}  loss {tot/steps:8.4f}  "
                      f"1-NFE  dmean {f['d_mean']:.4f}  dstd {f['d_std']:.4f}  "
                      f"mmd {score:.5f}{star}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=mf.state_dict(), ema=ema.shadow.state_dict(),
                            opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                            best=best), state_path)
    return ck, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='nvae', choices=['vae', 'nvae'])
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--variant', default='meanflow',
                    choices=['meanflow', 'meanflow_improved'])
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--sweep', action='store_true',
                    help='short hyper-parameter sweep for the improved variant')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--suffix', default='', help="stage-1 checkpoint suffix, e.g. _fine8")
    ap.add_argument('--z_fine_ch', type=int, default=None)
    a = ap.parse_args()

    Z_tr, Z_va = latent_dataset(a.model, a.dataset, a.suffix, a.z_fine_ch)
    print(f"latents train={tuple(Z_tr.shape)} val={tuple(Z_va.shape)}  "
          f"std {Z_tr.std():.4f}  per-dim [{Z_tr.std(0).min():.3f}, {Z_tr.std(0).max():.3f}]")

    if a.sweep:
        grid = [
            dict(p_ratio=0.75, p_boundary=0.20, adaptive_p=0.5),
            dict(p_ratio=0.50, p_boundary=0.25, adaptive_p=0.5),
            dict(p_ratio=0.50, p_boundary=0.25, adaptive_p=1.0),
            dict(p_ratio=0.25, p_boundary=0.30, adaptive_p=0.5),
        ]
        res = []
        ep = a.epochs
        for i, kw in enumerate(grid):
            print(f"\n--- sweep {i+1}/{len(grid)}: {kw} ---", flush=True)
            _, b = train('meanflow_improved', Z_tr, Z_va, a,
                         f"_sweep_{a.model}_{i}", mf_kw=kw)
            res.append((b, kw))
            print(f"    -> mmd {b:.5f}")
        res.sort(key=lambda r: r[0])
        print("\n=== sweep ranking (lower mmd better) ===")
        for b, kw in res:
            print(f"  {b:.5f}  {kw}")
        json.dump({'best': res[0][1], 'all': [[b, k] for b, k in res]},
                  open(f"checkpoints/_sweep_{a.model}_improved.json", 'w'), indent=2)
    else:
        tag = a.tag or f"{a.model}_{a.dataset}{a.suffix}_{a.variant}_stage2"
        kw = {}
        sp = f"checkpoints/_sweep_{a.model}_improved.json"
        if a.variant == 'meanflow_improved' and os.path.exists(sp):
            kw = json.load(open(sp))['best']
            print(f"using swept hyper-params: {kw}")
        print(f"\n=== training {a.variant} on {a.model} latents ===", flush=True)
        ck, best = train(a.variant, Z_tr, Z_va, a, tag, mf_kw=kw)
        print(f"best 1-NFE latent MMD {best:.5f} -> {ck}")
