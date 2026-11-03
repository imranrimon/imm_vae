"""VQ-VAE rung of the Stage-1 ladder: autoencoder, then an autoregressive prior.

Two stages, matching the continuous pipeline:
  --stage ae     encoder/decoder + codebook, same losses as the continuous AE
                 (L1 + LPIPS + PatchGAN) plus the VQ commitment term
  --stage prior  autoregressive transformer over the 8x8 code grid

Reported against the same three quantities as every other compressor: the
ceiling (decode ground-truth codes), the free baseline, and the learned prior.
For a discrete latent the free baseline is sampling each code independently
from its marginal frequency -- the discrete analogue of a factorized Gaussian.

Usage:
    python train_vq.py --stage ae --epochs 25
    python train_vq.py --stage prior --epochs 30
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import math
import time

import torch
import torch.nn.functional as F

from src.models.vq_ae import VqAutoencoder, CodeTransformer
from src.models.ldm_ae import PatchDiscriminator, hinge_d_loss, adaptive_weight
from train_stage1_fast import image_cache, to_float
from fid_standard import StandardFid, run

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def ae_tag(a):
    q = '' if a.quantizer == 'loss' else '_ema'
    ab = ''
    if a.quantizer == 'ema':
        ab += '' if a.l2 else '-nol2'
        ab += '' if a.restart else '-nores'
        ab += '' if a.kmeans else '-nokm'
    return f"vqae_{a.dataset}_z{a.z_ch}_c{a.n_codes}{q}{ab}"


def make_ae(a):
    kw = {}
    if a.quantizer == 'ema':
        kw = dict(l2_norm=a.l2, restart=a.restart, kmeans_init=a.kmeans)
    return VqAutoencoder(base=128, z_ch=a.z_ch, n_res=2, n_codes=a.n_codes,
                         quantizer=a.quantizer, **kw)


def train_ae(a):
    Xtr = image_cache(a.dataset, 'train', a.image_size).to(DEVICE)
    Xva = image_cache(a.dataset, 'test', a.image_size).to(DEVICE)
    S = a.image_size // 4
    print(f"latent: {S}x{S} codes from a {a.n_codes}-entry codebook "
          f"({S*S} tokens, {S*S*math.log2(a.n_codes)/8:.0f} bytes/image)")

    ae = make_ae(a).to(DEVICE)
    disc = PatchDiscriminator().to(DEVICE)
    import lpips
    percep = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for p in percep.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(ae.parameters(), lr=a.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr * 2, betas=(0.5, 0.9))

    ck, sp = f"checkpoints/{ae_tag(a)}.pth", f"checkpoints/{ae_tag(a)}_state.pth"
    start, best = 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        ae.load_state_dict(st['ae']); opt.load_state_dict(st['opt'])
        try:
            disc.load_state_dict(st['disc']); opt_d.load_state_dict(st['opt_d'])
        except Exception:
            print("  (reinitializing D)")
        start, best = st['epoch'], st['best']
        print(f"  resumed at epoch {start} (best {best:.4f})")

    n, bs = len(Xtr), a.bs
    for ep in range(start, a.epochs):
        ae.train(); disc.train()
        perm = torch.randperm(n, device=DEVICE)
        agg = dict(rec=0., perc=0., vq=0., used=0.)
        steps = 0
        use_gan = ep >= a.disc_start_ep
        t0 = time.time()
        for i in range(0, n - bs + 1, bs):
            x = to_float(Xtr[perm[i:i + bs]])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                x_rec, vq_loss, idx = ae(x)
                rec = (x - x_rec).abs().mean()
                perc = percep(x_rec, x).mean()
            nll = (rec + perc).float()
            loss = nll + vq_loss.float()

            if use_gan:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                    g_loss = -disc(x_rec).mean()
                g_loss = g_loss.float()
                try:
                    w = adaptive_weight(nll, g_loss, ae.last_layer())
                except RuntimeError:
                    w = torch.tensor(0.0, device=DEVICE)
                loss = loss + 0.5 * w * g_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()

            if use_gan:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                    d_loss = hinge_d_loss(disc(x), disc(x_rec.detach())).float()
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_d.step()

            agg['rec'] += float(rec); agg['perc'] += float(perc)
            agg['vq'] += float(vq_loss)
            agg['used'] += idx.unique().numel()
            steps += 1

        ae.eval()
        with torch.no_grad():
            vr = vp = vs = 0
            for j in range(0, min(len(Xva), 5120) - bs + 1, bs):
                x = to_float(Xva[j:j + bs])
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                    xr, _, _ = ae(x)
                    vr += float((x - xr).abs().mean())
                    vp += float(percep(xr, x).mean())
                vs += 1
            vr /= vs; vp /= vs
        val = vr + vp
        star = ''
        if val < best:
            best = val; torch.save(ae.state_dict(), ck); star = '  *'
        alive = (f" alive {ae.quant.usage()*100:.0f}%"
                 if hasattr(ae.quant, 'usage') else "")
        print(f"  ep {ep+1:3d}/{a.epochs}  L1 {agg['rec']/steps:.4f}  "
              f"LPIPS {agg['perc']/steps:.4f}  vq {agg['vq']/steps:.4f}  "
              f"codes/batch {agg['used']/steps:.0f}/{a.n_codes}{alive} | "
              f"val L1 {vr:.4f} LPIPS {vp:.4f}{star}  ({time.time()-t0:.0f}s)",
              flush=True)
        torch.save(dict(ae=ae.state_dict(), disc=disc.state_dict(),
                        opt=opt.state_dict(), opt_d=opt_d.state_dict(),
                        epoch=ep + 1, best=best), sp)


@torch.no_grad()
def extract_codes(ae, X, bs=512):
    out = []
    for i in range(0, len(X), bs):
        x = to_float(X[i:i + bs]).to(DEVICE)
        _, _, idx = ae(x)
        out.append(idx.reshape(len(x), -1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='ae', choices=['ae', 'prior'])
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--z_ch', type=int, default=4)
    ap.add_argument('--n_codes', type=int, default=1024)
    ap.add_argument('--quantizer', default='loss', choices=['loss', 'ema'])
    ap.add_argument('--no-l2', dest='l2', action='store_false')
    ap.add_argument('--no-restart', dest='restart', action='store_false')
    ap.add_argument('--no-kmeans', dest='kmeans', action='store_false')
    ap.add_argument('--tr_dim', type=int, default=256)
    ap.add_argument('--tr_layers', type=int, default=6)
    ap.add_argument('--tr_heads', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--bs', type=int, default=160)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--disc_start_ep', type=int, default=6)
    ap.add_argument('--amp', action='store_true', default=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval-only', dest='eval_only', action='store_true')
    ap.add_argument('--fake', type=int, default=50000)
    a = ap.parse_args()

    if a.stage == 'ae' and not a.eval_only:
        train_ae(a)
        return

    # ---------------- prior over codes ----------------
    ae = make_ae(a).to(DEVICE)
    ae.load_state_dict(torch.load(f"checkpoints/{ae_tag(a)}.pth", map_location=DEVICE))
    ae.eval()
    S = a.image_size // 4
    Xtr = image_cache(a.dataset, 'train', a.image_size)

    # must include the quantizer: the loss-based and EMA codebooks give
    # completely different code assignments, and a shared cache silently
    # evaluates one model's decoder against the other's codes.
    cpath = f"checkpoints/_vqcodes_{a.dataset}_{a.n_codes}_{a.quantizer}.pt"
    if os.path.exists(cpath):
        codes = torch.load(cpath)
    else:
        codes = extract_codes(ae, Xtr)
        torch.save(codes, cpath)
    print(f"codes {tuple(codes.shape)}  unique used: {codes.unique().numel()}/{a.n_codes}")

    tsz = '' if (a.tr_dim, a.tr_layers) == (256, 6) else f"_d{a.tr_dim}l{a.tr_layers}"
    tag = f"vqprior_{a.dataset}_c{a.n_codes}_{a.quantizer}{tsz}"
    ck, sp = f"checkpoints/{tag}.pth", f"checkpoints/{tag}_state.pth"
    tr = CodeTransformer(n_codes=a.n_codes, seq=S * S, dim=a.tr_dim,
                         heads=a.tr_heads, layers=a.tr_layers).to(DEVICE)
    print(f"code transformer: {sum(p.numel() for p in tr.parameters())/1e6:.2f}M "
          f"params (dim {a.tr_dim}, {a.tr_layers} layers)")
    opt = torch.optim.AdamW(tr.parameters(), lr=3e-4, betas=(0.9, 0.95))
    start, best = 0, float('inf')
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location=DEVICE)
        tr.load_state_dict(st['m']); opt.load_state_dict(st['opt'])
        start, best = st['epoch'], st['best']
        print(f"  resumed at epoch {start} (best nll {best:.4f})")

    if not a.eval_only:
        C = codes.to(DEVICE)
        n, bs = len(C), 256
        for ep in range(start, a.epochs):
            tr.train()
            perm = torch.randperm(n, device=DEVICE)
            tot = steps = 0
            t0 = time.time()
            for i in range(0, n - bs + 1, bs):
                loss = tr.loss(C[perm[i:i + bs]])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
                opt.step()
                tot += loss.item(); steps += 1
            nll = tot / steps
            star = ''
            if nll < best:
                best = nll; torch.save(tr.state_dict(), ck); star = '  *'
            print(f"  ep {ep+1:3d}/{a.epochs}  code NLL {nll:.4f} "
                  f"(uniform {math.log(a.n_codes):.4f}){star}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(m=tr.state_dict(), opt=opt.state_dict(),
                            epoch=ep + 1, best=best), sp)

    tr.load_state_dict(torch.load(ck, map_location=DEVICE)); tr.eval()

    # marginal code frequencies -> the free discrete baseline
    freq = torch.bincount(codes.reshape(-1), minlength=a.n_codes).float()
    freq = (freq / freq.sum()).to(DEVICE)

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    Ct = codes.to(DEVICE)
    cfg = [
        ('vq_ae', 'TRUE codes [ceiling]', 0,
         lambda b: ae.decode_codes(
             Ct[torch.randint(0, len(Ct), (b,), device=DEVICE)].view(b, S, S))),
        ('vq_ae', 'independent marginal codes', 0,
         lambda b: ae.decode_codes(
             torch.multinomial(freq, b * S * S, replacement=True).view(b, S, S))),
    ]
    # An autoregressive sampler's quality depends strongly on the truncation
    # used, so sweep it rather than reporting one arbitrary setting.
    for temp, tk in [(1.0, 0), (0.9, 0), (1.0, 100), (0.9, 100), (0.8, 50)]:
        cfg.append(('vq_ae', f'AR transformer T={temp} top_k={tk or "off"}', S * S,
                    lambda b, t=temp, k=tk: ae.decode_codes(
                        tr.sample(b, DEVICE, temperature=t, top_k=k).view(b, S, S))))
    df = run(cfg, a.fake, fid, f'results/{a.dataset}/vq.csv',
             variant=f'c{a.n_codes}-{a.quantizer}{tsz}'
                     + ('' if a.fake == 50000 else f'-n{a.fake}'))
    print("\n=== VQ-VAE ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
