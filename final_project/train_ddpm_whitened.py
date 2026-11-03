"""Many-step conditional DDPM on the whitened residual.

Purpose: decide whether "a learned prior cannot beat the closed-form baseline"
is a fact about LEARNED PRIORS or a fact about MEANFLOW.

Established so far on the whitened residual (target excess kurtosis +0.547):
  * MeanFlow output kurtosis is ~0.000 at every NFE from 1 to 64. The learned
    map is linear, so extra sampling steps cannot create non-Gaussianity.
  * Closed form: covariance -> FID 41.65, covariance+copula -> 40.97.

A DDPM trained with the standard epsilon-prediction objective and sampled over
many steps is a genuinely different function class -- the iterative denoiser is
nonlinear in a way a one-step regression map is not. If this recovers the
kurtosis, the limitation was MeanFlow's objective. If it does not, the
limitation is the target or the conditioning.

Usage:
    python train_ddpm_whitened.py --epochs 60
    python train_ddpm_whitened.py --eval-only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch

from src.models.meanflow import MeanFlowNet
from train_meanflow import Ema, DEVICE
from train_joint_prior import load_base, joint_latents, decode_joint, Z_G, Z_F
from train_residual_fine_prior import decoder_fine_prior
from train_whitened_prior import whitening, nongauss

TAG = 'nvae_CelebA_fine8_whitened_ddpm'


def cosine_alphas(T, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-8, 1.0)
    betas = (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)
    return betas


class CondDDPM(torch.nn.Module):
    """eps-prediction DDPM over a flat latent, conditioned on z_global."""

    def __init__(self, D=Z_F, cond_dim=Z_G, width=768, blocks=6, T=1000):
        super().__init__()
        self.D, self.T = D, T
        # MeanFlowNet already accepts (z, r, t, cond); pass r == t for a plain denoiser
        self.net = MeanFlowNet(z_dim=D, model_dim=width, n_blocks=blocks,
                               cond_dim=cond_dim)
        betas = cosine_alphas(T)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1 - betas, dim=0))

    def eps(self, z, t_norm, cond):
        return self.net(z, t_norm, t_norm, cond)

    def compute_loss(self, x, cond):
        b = x.shape[0]
        t = torch.randint(0, self.T, (b,), device=x.device)
        ab = self.alphas_cumprod[t].unsqueeze(1)
        noise = torch.randn_like(x)
        z = ab.sqrt() * x + (1 - ab).sqrt() * noise
        tn = (t.float() / self.T).unsqueeze(1)
        return ((self.eps(z, tn, cond) - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond, steps=100, eta=0.0, x0_clip=5.0):
        """DDIM sampler; eta=0 is deterministic, eta=1 is ancestral.

        x0 is clipped because the cosine schedule drives alpha_bar_T to ~1e-8,
        so the 1/sqrt(alpha_bar) division in the x0 estimate amplifies by ~1e4
        and the trajectory diverges. The target here is unit-scale, so +-5 sigma
        is generous.
        """
        b = cond.shape[0]
        z = torch.randn(b, self.D, device=cond.device)
        seq = torch.linspace(self.T - 1, 0, steps).long().to(cond.device)
        for i in range(len(seq)):
            t = seq[i]
            ab_t = self.alphas_cumprod[t]
            ab_p = self.alphas_cumprod[seq[i + 1]] if i + 1 < len(seq) \
                else torch.tensor(1.0, device=cond.device)
            tn = torch.full((b, 1), float(t) / self.T, device=cond.device)
            e = self.eps(z, tn, cond)
            x0 = (z - (1 - ab_t).sqrt() * e) / ab_t.sqrt().clamp(min=1e-3)
            if x0_clip:
                x0 = x0.clamp(-x0_clip, x0_clip)
            sig = eta * ((1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)).clamp(min=0).sqrt()
            dir_z = (1 - ab_p - sig ** 2).clamp(min=0).sqrt() * e
            z = ab_p.sqrt() * x0 + dir_z
            if eta > 0 and i + 1 < len(seq):
                z = z + sig * torch.randn_like(z)
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
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
    print(f"whitened target: std {Utr.std():.4f}  excess kurtosis {nongauss(Uva):+.4f}")

    model = CondDDPM(width=a.width, blocks=a.blocks).to(DEVICE)
    ema = Ema(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99))

    ck, sp = f"checkpoints/{TAG}.pth", f"checkpoints/{TAG}_train_state.pth"
    start, gstep, best = 0, 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        model.load_state_dict(st['model']); ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start, gstep, best = st['epoch'], st['gstep'], st['best']
        print(f"  resumed at epoch {start} (best kurt gap {best:.4f})")

    n, bs = len(Utr), a.bs
    total = (n // bs) * a.epochs
    warm = min(500, total // 10)

    def lr_at(g):
        if g < warm:
            return g / max(warm, 1)
        p = (g - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    if not a.eval_only:
        print(f"\n=== conditional DDPM on whitened residual (T=1000) ===", flush=True)
        t0 = time.time()
        for ep in range(start, a.epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            tot = steps = 0
            for i in range(0, n - bs + 1, bs):
                j = perm[i:i + bs]
                for g in opt.param_groups:
                    g['lr'] = a.lr * lr_at(gstep)
                loss = model.compute_loss(Utr[j], Ztr[j][:, :Z_G])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ema.update(model)
                tot += loss.item(); steps += 1; gstep += 1

            if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
                with torch.no_grad():
                    cond = Zva[torch.randperm(len(Zva), device=DEVICE)[:4096]][:, :Z_G]
                    u = ema.shadow.sample(cond, steps=100, eta=1.0)
                k, kt = nongauss(u), nongauss(Uva)
                gap = abs(k - kt) + 3.0 * abs(float(u.std()) - 1.0)
                star = ''
                if gap < best:
                    best = gap; torch.save(ema.shadow.state_dict(), ck); star = '  *'
                print(f"  [ddpm] ep {ep+1:3d}/{a.epochs}  loss {tot/steps:.5f}  "
                      f"std {u.std():.4f}  kurt {k:+.3f} (true {kt:+.3f})  "
                      f"gap {gap:.4f}{star}  ({time.time()-t0:.0f}s)", flush=True)
                torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(),
                                opt=opt.state_dict(), epoch=ep + 1, gstep=gstep,
                                best=best), sp)

    model.load_state_dict(torch.load(ck, map_location=DEVICE)); model.eval()
    print("\nstep sweep -- does a many-step denoiser recover the non-Gaussianity?")
    with torch.no_grad():
        cond = Zva[torch.randperm(len(Zva), device=DEVICE)[:8192]][:, :Z_G]
        for st in [10, 50, 100, 250]:
            for eta in [0.0, 1.0]:
                u = model.sample(cond, steps=st, eta=eta)
                print(f"  steps {st:4d} eta {eta:.0f} : std {u.std():.4f}  "
                      f"kurt {nongauss(u):+.4f}  (target {nongauss(Uva):+.4f})", flush=True)

    from fid_standard import StandardFid, run
    fid = StandardFid(n_real=50000)
    cfg = []

    def mk(sg, st, eta):
        def f(b):
            zg = sg * torch.randn(b, Z_G, device=DEVICE)
            mp, sp_ = decoder_fine_prior(base, zg)
            u = model.sample(zg, steps=st, eta=eta)
            return decode_joint(base, torch.cat([zg, mp + sp_ * (mu + u @ L.T)], 1))
        return f
    for sg in [1.0, 1.4]:
        for st in [50, 250]:
            cfg.append(('nvae_fine8', f'z_g~N(0,{sg}) + WHITENED DDPM (eta=1)', st,
                        mk(sg, st, 1.0)))
    print()
    df = run(cfg, a.fake, fid, 'results/CelebA/whitened_ddpm.csv')
    print("\n=== MANY-STEP DDPM vs CLOSED FORM (floor 41.65 / copula 40.97) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
