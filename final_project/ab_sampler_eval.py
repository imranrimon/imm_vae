"""A/B evaluation of the IMM latent sampler on CelebA.

Holds the compressor, prior weights, FID feature extractor, sample counts and
RNG seed fixed, and varies only the sampler:

  legacy : ascending time grid (t < s) + the transition applied twice per step
  fixed  : descending time grid (t > s, the domain compute_loss trains on),
           transition applied once per step

Baselines (Gaussian prior, latent DDPM) are included unchanged for context.

Note on the NVAE IMM checkpoint: it predates the data_mean/data_std buffers and
was trained on raw (unnormalized) latents, so it is loaded with strict=False and
identity normalization, which is the regime it was trained under.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights

from src.data import get_dataloader
from src.models.vae import VAE
from src.models.nvae import NVAE
from src.models.diffusion import DDPM, ResidualDenoiser
from src.models.imm_core import IMM, OtFmEulerFmParameterization
from src.models.meanflow import MeanFlow, MeanFlowNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_DIM = 64
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# ---------------------------------------------------------------- samplers
@torch.no_grad()
def imm_sample(model, n, steps, mode):
    """mode='fixed' -> corrected sampler; mode='legacy' -> the original buggy one."""
    if mode == 'fixed':
        return model.sample(n, model.sampling_times(steps, device=DEVICE))

    # legacy: ascending grid, transition applied twice (verbatim original behaviour)
    times = torch.linspace(0, 1, steps + 1, device=DEVICE)
    x = model.parameterization.sigma_data * torch.randn((n, model.D), device=DEVICE)
    for i in range(len(times) - 1):
        t_tensor = torch.full((n, 1), times[i], device=DEVICE)
        s_tensor = torch.full((n, 1), times[i + 1], device=DEVICE)
        x = model.forward(x, s_tensor, t_tensor)
        x = model.forward(x, s_tensor, t_tensor)
    return x * model.data_std + model.data_mean


# ---------------------------------------------------------------- models
def load_base(model_type, method):
    ckpt = f"checkpoints/{model_type}_CelebA_{method}_stage1.pth"
    m = VAE(latent_dim=Z_DIM) if model_type == 'vae' else NVAE(z_dim=Z_DIM)
    m = m.to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.eval()


def load_prior(model_type, method):
    ckpt = f"checkpoints/{model_type}_CelebA_{method}_stage2.pth"
    sd = torch.load(ckpt, map_location=DEVICE)
    if method == 'ddpm':
        net = ResidualDenoiser(input_dim=Z_DIM + 64, model_dim=512, output_dim=Z_DIM, n_blocks=6)
        m = DDPM(net, z_dim=Z_DIM, n_steps=1000, device=DEVICE).to(DEVICE)
        m.load_state_dict(sd)
        return m.eval(), True
    net = ResidualDenoiser(input_dim=Z_DIM + 2, model_dim=512, output_dim=Z_DIM, n_blocks=6)
    m = IMM(net, OtFmEulerFmParameterization(sigma_data=1.0), D=Z_DIM, M=16).to(DEVICE)
    normed = 'data_mean' in sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    if not normed:
        print(f"  ! {model_type} IMM ckpt has no normalization buffers "
              f"(missing={list(missing)}); using identity, matching its training regime.")
        m.data_mean.zero_()
        m.data_std.fill_(1.0)
    return m.eval(), normed


@torch.no_grad()
def decode(base, model_type, z):
    if model_type == 'vae':
        return base.decode(z)
    out = base.decoder(z, z_fine=None)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def gen_images(n, model_type, base, sampler, bs=500):
    out = []
    done = 0
    while done < n:
        b = min(bs, n - done)
        out.append(decode(base, model_type, sampler(b)).cpu())
        done += b
    return torch.cat(out)[:n]


# ---------------------------------------------------------------- FID
class Fid:
    def __init__(self):
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           transform_input=False).to(DEVICE)
        net.fc = nn.Identity()
        self.net = net.eval()
        self._real = {}

    @torch.no_grad()
    def feats(self, imgs, bs=100):
        out = []
        for i in range(0, len(imgs), bs):
            b = imgs[i:i + bs].to(DEVICE) * 0.5 + 0.5
            b = F.interpolate(b, size=(299, 299), mode='bilinear', align_corners=False)
            b = (b - IMAGENET_MEAN) / IMAGENET_STD
            f = self.net(b)
            if hasattr(f, 'logits'):
                f = f.logits
            out.append(f.cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def real_stats(self, n):
        if n in self._real:
            return self._real[n]
        cache = f"results/CelebA/_real_feats_{n}.npz"
        if os.path.exists(cache):
            d = np.load(cache)
            self._real[n] = (d['mu'], d['sigma'])
            print(f"  real reference: {n} images (cached)")
            return self._real[n]
        loader = get_dataloader('CelebA', batch_size=100, split='test', drop_last=False)
        fs, cnt = [], 0
        for x, _ in loader:
            fs.append(self.feats(x))
            cnt += x.shape[0]
            if cnt >= n:
                break
        f = np.concatenate(fs)[:n]
        mu, sigma = f.mean(0), np.cov(f, rowvar=False)
        os.makedirs('results/CelebA', exist_ok=True)
        np.savez(cache, mu=mu, sigma=sigma)
        self._real[n] = (mu, sigma)
        print(f"  real reference: {f.shape[0]} images")
        return self._real[n]

    def score(self, imgs, n_real):
        mu_r, sg_r = self.real_stats(n_real)
        f = self.feats(imgs)
        mu_f, sg_f = f.mean(0), np.cov(f, rowvar=False)
        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sg_r.dot(sg_f), disp=False)
        if not np.isfinite(covmean).all():
            off = np.eye(sg_r.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sg_r + off).dot(sg_f + off))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sg_r) + np.trace(sg_f) - 2 * np.trace(covmean))


# ---------------------------------------------------------------- driver
def run(model_type, n_fake, n_real, fid, out_csv, seed=0):
    rows = []
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict('records')
        done = {(str(r['Backbone']), str(r['Method']), str(r['Steps']), str(r['Sampler']))
                for r in rows}

    def record(method, steps, sampler_name, nfe, base, sampler):
        key = (str(model_type), str(method), str(steps), str(sampler_name))
        if key in done:
            print(f"  [{model_type}] {method:11s} steps={str(steps):>4s} "
                  f"{sampler_name:6s} -- cached, skipped", flush=True)
            return
        torch.manual_seed(seed)
        imgs = gen_images(n_fake, model_type, base, sampler)
        v = fid.score(imgs, n_real)
        rows.append(dict(Backbone=model_type, Method=method, Steps=steps,
                         Sampler=sampler_name, NFE=nfe, FID=round(v, 4)))
        done.add(key)
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # checkpoint every row
        print(f"  [{model_type}] {method:11s} steps={str(steps):>4s} "
              f"{sampler_name:6s} NFE={str(nfe):>4s}  FID={v:.4f}", flush=True)

    for m in ['kl', 'mmd']:
        base = load_base(model_type, m)
        record(f"{m.upper()}-Base", '-', '-', 0, base,
               lambda b: torch.randn(b, Z_DIM, device=DEVICE))

    base_kl = load_base(model_type, 'kl')

    ddpm, _ = load_prior(model_type, 'ddpm')
    for s in [100, 500, 1000]:
        record('DDPM', s, 'orig', s, base_kl, lambda b, s=s: ddpm.sample(b, steps=s))

    imm, normed = load_prior(model_type, 'imm')
    print(f"  IMM normalization buffers present: {normed}")
    for s in [8, 16, 32, 64]:
        for mode in ['legacy', 'fixed']:
            nfe = 2 * s if mode == 'legacy' else s
            record('IMM', s, mode, nfe, base_kl,
                   lambda b, s=s, mo=mode: imm_sample(imm, b, s, mo))

    # Control: is the "learned prior" gain just a latent temperature rescale?
    # The KL base's z_global is already close to N(0, I), and the learned samplers
    # mostly change its scale -- so sweep sigma on a plain Gaussian and see how
    # much of the gain a single scalar buys.
    for sig in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        record('Gaussian-T', f"sigma={sig}", 'temp', 0, base_kl,
               lambda b, sg=sig: sg * torch.randn(b, Z_DIM, device=DEVICE))

    # MeanFlow priors (standard convention, genuine 1-NFE sampling)
    for variant, label in [('meanflow', 'MeanFlow'),
                           ('meanflow_improved', 'MeanFlow-I')]:
        ck = f"checkpoints/{model_type}_CelebA_{variant}_stage2.pth"
        if not os.path.exists(ck):
            print(f"  (no {ck}, skipping {label})")
            continue
        net = MeanFlowNet(z_dim=Z_DIM, model_dim=512, n_blocks=6)
        mf = MeanFlow(net, D=Z_DIM,
                      variant='base' if variant == 'meanflow' else 'improved').to(DEVICE)
        mf.load_state_dict(torch.load(ck, map_location=DEVICE))
        mf.eval()
        for st in [1, 2, 4, 8, 16]:
            record(label, st, 'mf', st, base_kl,
                   lambda b, st=st, m=mf: m.sample(b, steps=st))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fake', type=int, default=5000)
    ap.add_argument('--real', type=int, default=10000)
    ap.add_argument('--models', nargs='+', default=['vae', 'nvae'])
    ap.add_argument('--out', default='results/CelebA/ab_sampler.csv')
    a = ap.parse_args()

    print(f"device={DEVICE}  fake={a.fake}  real={a.real}", flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fid = Fid()
    for mt in a.models:
        print(f"\n=== {mt.upper()} / CelebA ===", flush=True)
        run(mt, a.fake, a.real, fid, a.out)

    df = pd.read_csv(a.out)
    print("\n=== A/B RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {a.out}")
