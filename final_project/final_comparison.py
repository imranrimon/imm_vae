"""Matched-NFE comparison of every latent prior, on FID *and* latent fidelity.

FID alone is misleading here: on this NVAE the global latent is already close to
N(0, I), and FID rewards inflating its variance. So every row also reports how
well the prior actually reproduces the held-out encoder latent distribution
(per-dim mean/std error and RBF-MMD). NFE is the true number of network
evaluations, not the nominal "steps".
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import pandas as pd
import torch

from ab_sampler_eval import Fid, load_base, gen_images, DEVICE, Z_DIM, imm_sample
from src.models.imm_core import (IMM, OtFmEulerFmParameterization,
                                 OtFmEulerFmParameterizationFixed)
from src.models.diffusion import ResidualDenoiser, DDPM
from src.models.meanflow import MeanFlow, MeanFlowNet
from train_meanflow import latent_dataset, rbf_mmd


def mf_load(model_type, variant):
    ck = f"checkpoints/{model_type}_CelebA_{variant}_stage2.pth"
    if not os.path.exists(ck):
        return None
    net = MeanFlowNet(z_dim=Z_DIM, model_dim=512, n_blocks=6)
    m = MeanFlow(net, D=Z_DIM,
                 variant='base' if variant == 'meanflow' else 'improved').to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE))
    return m.eval()


def imm_load(model_type, convention):
    if convention == 'fixed':
        ck = f"checkpoints/{model_type}_CelebA_imm_fixed_stage2.pth"
        param = OtFmEulerFmParameterizationFixed(sigma_data=1.0)
    else:
        ck = f"checkpoints/{model_type}_CelebA_imm_stage2.pth"
        param = OtFmEulerFmParameterization(sigma_data=1.0)
    if not os.path.exists(ck):
        return None
    net = ResidualDenoiser(input_dim=Z_DIM + 2, model_dim=512, output_dim=Z_DIM, n_blocks=6)
    m = IMM(net, param, D=Z_DIM, M=16, convention=convention).to(DEVICE)
    sd = torch.load(ck, map_location=DEVICE)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, unexp
    if 'data_mean' not in sd:
        m.data_mean.zero_(); m.data_std.fill_(1.0)
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='nvae')
    ap.add_argument('--fake', type=int, default=5000)
    ap.add_argument('--real', type=int, default=10000)
    ap.add_argument('--out', default='results/CelebA/final_comparison.csv')
    a = ap.parse_args()

    base = load_base(a.model, 'kl')
    fid = Fid()
    _, Zva = latent_dataset(a.model, 'CelebA')
    Zva = Zva.to(DEVICE)

    rows = []
    if os.path.exists(a.out):
        rows = pd.read_csv(a.out).to_dict('records')
    done = {(r['Prior'], r['NFE']) for r in rows}

    def add(prior, nfe, sampler):
        if (prior, nfe) in done:
            print(f"  {prior:14s} NFE={nfe:>4}  cached"); return
        torch.manual_seed(0)
        with torch.no_grad():
            zg = sampler(8192)
        r = Zva[torch.randperm(len(Zva), device=DEVICE)[:8192]]
        d_std = (zg.std(0) - r.std(0)).abs().mean().item()
        d_mean = (zg.mean(0) - r.mean(0)).abs().mean().item()
        mmd = rbf_mmd(zg, r)
        torch.manual_seed(0)
        imgs = gen_images(a.fake, a.model, base, sampler)
        f = fid.score(imgs, a.real)
        rows.append(dict(Prior=prior, NFE=nfe, FID=round(f, 3),
                         d_mean=round(d_mean, 4), d_std=round(d_std, 4),
                         latent_MMD=round(mmd, 6)))
        done.add((prior, nfe))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"  {prior:14s} NFE={nfe:>4}  FID={f:7.3f}  dmean={d_mean:.4f}  "
              f"dstd={d_std:.4f}  latMMD={mmd:.6f}", flush=True)

    add('Gaussian', 0, lambda b: torch.randn(b, Z_DIM, device=DEVICE))
    add('Gaussian-T1.5', 0, lambda b: 1.5 * torch.randn(b, Z_DIM, device=DEVICE))

    ddpm = DDPM(ResidualDenoiser(input_dim=Z_DIM + 64, model_dim=512,
                                 output_dim=Z_DIM, n_blocks=6),
                z_dim=Z_DIM, n_steps=1000, device=DEVICE).to(DEVICE)
    ddpm.load_state_dict(torch.load(
        f"checkpoints/{a.model}_CelebA_ddpm_stage2.pth", map_location=DEVICE))
    ddpm.eval()
    for s in [100, 1000]:
        add('DDPM', s, lambda b, s=s: ddpm.sample(b, steps=s))

    legacy = imm_load(a.model, 'legacy')
    if legacy is not None:
        for s in [8, 16, 32]:
            add('IMM-legacy', 2 * s, lambda b, s=s: imm_sample(legacy, b, s, 'legacy'))

    fixed = imm_load(a.model, 'fixed')
    if fixed is not None:
        for s in [1, 2, 4, 8, 16]:
            add('IMM-fixed', s,
                lambda b, s=s: fixed.sample(b, fixed.sampling_times(s, device=DEVICE)))

    for variant, label in [('meanflow', 'MeanFlow'), ('meanflow_improved', 'MeanFlow-I')]:
        m = mf_load(a.model, variant)
        if m is None:
            continue
        for s in [1, 2, 4, 8, 16]:
            add(label, s, lambda b, s=s, m=m: m.sample(b, steps=s))

    df = pd.DataFrame(rows).sort_values(['Prior', 'NFE'])
    print("\n=== FINAL COMPARISON (CelebA / NVAE) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {a.out}")
