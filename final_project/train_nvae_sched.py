"""Scheduled sampling for the NVAE hierarchy.

Both Stage-2 priors we tried failed for the same reason: each conditional prior
p(z_l | z_<l) is trained only on contexts z_<l drawn from the POSTERIOR, but at
sampling time z_<l comes from the PRIOR. By the fine groups the context has
drifted off the posterior manifold and every downstream conditional is evaluated
out of distribution. Fitting a better prior on posterior-derived pairs cannot
help, because those are not the pairs it meets at sampling time.

Scheduled sampling attacks that directly: during fine-tuning, each group's
context is replaced by a sample from its own prior with probability p, so the
downstream priors and the decoder are trained on the contexts they will actually
receive. The reconstruction target stays the real image, which is what makes the
signal meaningful -- the model must still produce that image even when part of
the hierarchy has been sampled rather than inferred.

The obvious cost is that reconstruction from a partly-sampled context is
genuinely harder, so p trades ceiling against sampling quality. We ramp p and
report both.

Usage:
    python train_nvae_sched.py --p 0.3 --epochs 20
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time

import torch
import torch.nn.functional as F

from src.models.nvae_proper import ProperNVAE
from train_stage1_fast import image_cache, to_float
from fid_standard import StandardFid, run

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def forward_sched(m, x, p):
    """NVAE forward where each group's CONTEXT is prior-sampled w.p. p.

    The KL is still computed from the posterior residual, so the variational
    objective is unchanged; only which sample gets folded into the top-down
    state changes.
    """
    feats = m._encode(x)
    B = x.shape[0]
    h = m.h0.expand(B, -1, -1, -1)
    kls, gi = [], 0
    for s in range(m.scales):
        ef = feats[s]
        for _ in range(m.gps_list[s]):
            mu_p, lv_p = m.prior[gi](h).chunk(2, 1)
            lv_p = lv_p.clamp(-8, 8)
            dmu, dlv = m.post[gi](torch.cat([h, ef], 1)).chunk(2, 1)
            dlv = dlv.clamp(-8, 8)
            mu_q = mu_p + dmu * torch.exp(lv_p)
            lv_q = lv_p + dlv
            z_q = mu_q + torch.randn_like(mu_q) * torch.exp(lv_q)
            kls.append(m._kl(dmu, dlv, lv_p).flatten(1).sum(1))

            if p > 0:
                z_p = mu_p + torch.randn_like(mu_p) * torch.exp(lv_p)
                use_prior = (torch.rand(B, 1, 1, 1, device=x.device) < p).float()
                z = use_prior * z_p + (1 - use_prior) * z_q
            else:
                z = z_q
            h = m.cells[gi](m.merge[gi](torch.cat([h, z], 1)))
            gi += 1
        if s < m.scales - 1:
            h = m.up[s](h)
    return torch.tanh(m.out(h)), kls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--z_ch', type=int, default=1)
    ap.add_argument('--scales', type=int, default=3)
    ap.add_argument('--gps', default='4,0,0')
    ap.add_argument('--norm', default='batch')
    ap.add_argument('--p', type=float, default=0.3, help='context-replacement rate')
    ap.add_argument('--p_warmup', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--free_bits', type=float, default=0.05)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    a = ap.parse_args()

    gps = [int(v) for v in a.gps.split(',')] if ',' in a.gps else int(a.gps)
    gtag = a.gps.replace(',', '')
    nm = '' if a.norm == 'batch' else '_gn'
    base_tag = f"nvaeP_{a.dataset}_s{a.scales}g{gtag}z{a.z_ch}{nm}"

    m = ProperNVAE(base=a.base, z_ch=a.z_ch, scales=a.scales,
                   groups_per_scale=gps, img=a.image_size, norm=a.norm).to(DEVICE)
    m.load_state_dict(torch.load(f"checkpoints/{base_tag}.pth", map_location=DEVICE))
    npix = a.image_size ** 2 * 3
    print(f"fine-tuning {base_tag}: {m.n_groups} groups, {m.latent_dims()} dims, "
          f"context replacement p={a.p}")

    tag = f"{base_tag}_sched{a.p}"
    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_state.pth"
    opt = torch.optim.Adamax(m.parameters(), lr=a.lr, weight_decay=3e-4)
    start, best = 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        m.load_state_dict(st['m']); opt.load_state_dict(st['opt'])
        start, best = st['epoch'], st['best']
        print(f"  resumed at epoch {start}")

    Xtr = image_cache(a.dataset, 'train', a.image_size).to(DEVICE)
    Xva = image_cache(a.dataset, 'test', a.image_size).to(DEVICE)
    kl_dims, r0 = [], a.image_size // 2 ** (a.scales - 1)
    for s_ in range(a.scales):
        for _ in range(m.gps_list[s_]):
            kl_dims.append(a.z_ch * r0 * r0)
        r0 *= 2

    n, bs = len(Xtr), a.bs
    if not a.eval_only:
        for ep in range(start, a.epochs):
            pp = a.p * min(1.0, (ep + 1) / a.p_warmup) if a.p_warmup else a.p
            m.train()
            perm = torch.randperm(n, device=DEVICE)
            tr = steps = 0
            t0 = time.time()
            for i in range(0, n - bs + 1, bs):
                x = to_float(Xtr[perm[i:i + bs]])
                rec, kls = forward_sched(m, x, pp)
                mse = F.mse_loss(rec, x)
                r = 0.5 * npix * torch.log(mse + 1e-8)
                kl = 0
                for k, d in zip(kls, kl_dims):
                    kl = kl + torch.clamp(k.mean() / d, min=a.free_bits) * d
                loss = r + kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 200.0)
                opt.step()
                tr += float(r); steps += 1

            m.eval()
            with torch.no_grad():
                vr = vs = 0
                for j in range(0, min(len(Xva), 5120) - bs + 1, bs):
                    x = to_float(Xva[j:j + bs])
                    rec, _ = m(x)
                    vr += float(F.mse_loss(rec, x, reduction='sum') / x.shape[0])
                    vs += 1
                vr /= vs
            star = ''
            if vr < best:
                best = vr; torch.save(m.state_dict(), ck); star = '  *'
            print(f"  ep {ep+1:3d}/{a.epochs}  p {pp:.2f}  recon {tr/steps:9.2f} | "
                  f"val MSE255 {vr/npix*(127.5**2):7.3f}{star}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(m=m.state_dict(), opt=opt.state_dict(),
                            epoch=ep + 1, best=best), sp)

    m.load_state_dict(torch.load(ck, map_location=DEVICE)); m.eval()
    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)

    def recon(b):
        idx = torch.randint(0, len(Xva), (b,), device=DEVICE)
        with torch.no_grad():
            r, _ = m(to_float(Xva[idx]))
        return r

    cfg = [('nvae_proper', f'sched p={a.p}: reconstructions [ceiling]', 0, recon)]
    for tt in [0.8, 1.0]:
        cfg.append(('nvae_proper', f'sched p={a.p}: ancestral t={tt}', 0,
                    lambda b, x=tt: m.sample(b, DEVICE, t=x)))
    import copy
    for tt in [0.8]:
        mr = copy.deepcopy(m).recalibrate_bn(n=20000, bs=250, device=DEVICE, t=tt)
        cfg.append(('nvae_proper', f'sched p={a.p}: ancestral t={tt} + BN recal', 0,
                    lambda b, mm=mr, x=tt: mm.sample(b, DEVICE, t=x)))
    df = run(cfg, 50000, fid, f'results/{a.dataset}/nvae_proper.csv',
             variant=f's{a.scales}g{gtag}z{a.z_ch}{nm}-sched{a.p}')
    print(f"\n=== SCHEDULED SAMPLING p={a.p} (baseline: ceiling 11.79, "
          f"best sampling 47.59) ===")
    print(df.tail(4).to_string(index=False))


if __name__ == '__main__':
    main()
