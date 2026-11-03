"""Precision / Recall / Density / Coverage on Inception features.

FID is a single number over a Gaussian fit to feature space, so it cannot
separate "samples look right" from "samples cover the data". This project's
central claim is that FID can be gamed by inflating latent variance, and that
claim cannot be supported with FID alone -- if a variance-inflated prior wins on
FID, it should lose on a fidelity/diversity split.

Implements the improved precision/recall of Kynkaanniemi et al. (2019) and the
density/coverage of Naeem et al. (2020), both k-NN manifold estimates on the
same 2048-d Inception pool features used for FID:

  precision  fraction of GENERATED samples inside the real manifold
  recall     fraction of REAL samples inside the generated manifold
  density    like precision but counts how many real balls a fake lands in,
             so it is not saturated by a few dense real regions
  coverage   fraction of real samples with >=1 generated neighbour in their
             own k-NN ball -- the most robust of the four to outliers

Usage:
    python eval_prdc.py --stack old
    python eval_prdc.py --stack new --dataset CelebA
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import pandas as pd
import torch

from fid_standard import StandardFid, DEVICE

K = 5           # Naeem et al. use k=5
N_SAMPLES = 10000   # standard for PRDC; 50k x 50k pairwise is not tractable


def _knn_radii(X, k, chunk=2000):
    """Distance from each row of X to its k-th nearest OTHER row."""
    n = len(X)
    out = torch.empty(n, device=X.device)
    for i in range(0, n, chunk):
        d = torch.cdist(X[i:i + chunk], X)
        # self-distance is 0, so take the (k+1)-th smallest
        out[i:i + chunk] = d.kthvalue(k + 1, dim=1).values
    return out


def _min_dist_and_counts(A, B, radii_B, chunk=2000):
    """For each row of A: min distance to B, and how many B-balls contain it."""
    n = len(A)
    mind = torch.empty(n, device=A.device)
    cnt = torch.zeros(n, device=A.device)
    for i in range(0, n, chunk):
        d = torch.cdist(A[i:i + chunk], B)
        mind[i:i + chunk] = d.min(dim=1).values
        cnt[i:i + chunk] = (d < radii_B.unsqueeze(0)).sum(dim=1).float()
    return mind, cnt


@torch.no_grad()
def prdc(real_feat, fake_feat, k=K):
    R = torch.as_tensor(real_feat, device=DEVICE, dtype=torch.float32)
    F = torch.as_tensor(fake_feat, device=DEVICE, dtype=torch.float32)
    rR = _knn_radii(R, k)
    rF = _knn_radii(F, k)

    # precision / density: where do fakes sit relative to real balls?
    _, cnt_f_in_r = _min_dist_and_counts(F, R, rR)
    precision = (cnt_f_in_r > 0).float().mean().item()
    density = (cnt_f_in_r.sum() / (k * len(F))).item()

    # recall: do reals sit inside the generated manifold?
    _, cnt_r_in_f = _min_dist_and_counts(R, F, rF)
    recall = (cnt_r_in_f > 0).float().mean().item()

    # coverage: does each real have a generated neighbour in its OWN ball?
    d_r_to_f, _ = _min_dist_and_counts(R, F, rF)
    coverage = (d_r_to_f < rR).float().mean().item()

    return dict(precision=round(precision, 4), recall=round(recall, 4),
                density=round(density, 4), coverage=round(coverage, 4))


def evaluate(configs, fid, n, out_csv):
    """configs: list of (label, make_batch). Reuses fid.features / fid.reference."""
    # real features from the same cache FID uses
    X = torch.load(f"data/_{fid.dataset}{fid.image_size}_train.pt")
    g = torch.Generator().manual_seed(1234)
    idx = torch.randperm(len(X), generator=g)[:n]
    real_feat = fid.features(X[idx])

    rows = []
    if os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict('records')
    done = {r['Prior'] for r in rows}

    for label, mk in configs:
        if label in done:
            print(f"  {label:44s} cached")
            continue
        torch.manual_seed(0)
        feats, got = [], 0
        while got < n:
            b = min(250, n - got)
            with torch.no_grad():
                imgs = mk(b)
            feats.append(fid.features(imgs.cpu()))
            got += b
            del imgs
        fake_feat = np.concatenate(feats)[:n]
        m = prdc(real_feat, fake_feat)
        m['Prior'] = label
        rows.append(m)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  {label:44s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"D={m['density']:.3f}  C={m['coverage']:.3f}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stacks
def old_stack_configs():
    """The FID-gaming test: temperature vs a prior that matches q(z) well."""
    from ab_sampler_eval import load_base, imm_sample, decode as dec
    from src.models.imm_core import IMM, OtFmEulerFmParameterization
    from src.models.diffusion import ResidualDenoiser
    from src.models.meanflow import MeanFlow, MeanFlowNet
    Z = 64
    base = load_base('nvae', 'kl')
    cfg = [('Gaussian sigma=1.0 [FID 42.2]',
            lambda b: dec(base, 'nvae', torch.randn(b, Z, device=DEVICE))),
           ('Temperature sigma=1.5 [FID 31.1]',
            lambda b: dec(base, 'nvae', 1.5 * torch.randn(b, Z, device=DEVICE)))]

    net = ResidualDenoiser(input_dim=Z + 2, model_dim=512, output_dim=Z, n_blocks=6)
    legacy = IMM(net, OtFmEulerFmParameterization(1.0), D=Z, M=16,
                 convention='legacy').to(DEVICE)
    sd = torch.load('checkpoints/nvae_CelebA_imm_stage2.pth', map_location=DEVICE)
    legacy.load_state_dict(sd, strict=False)
    if 'data_mean' not in sd:
        legacy.data_mean.zero_(); legacy.data_std.fill_(1.0)
    legacy.eval()
    cfg.append(('IMM-legacy, paper headline [FID 26.4]',
                lambda b: dec(base, 'nvae', imm_sample(legacy, b, 8, 'legacy'))))

    p = 'checkpoints/nvae_CelebA_meanflow_stage2.pth'
    if os.path.exists(p):
        mf = MeanFlow(MeanFlowNet(z_dim=Z, model_dim=512, n_blocks=6),
                      D=Z, variant='base').to(DEVICE)
        mf.load_state_dict(torch.load(p, map_location=DEVICE)); mf.eval()
        cfg.append(('MeanFlow, best latent fit [FID 40.3]',
                    lambda b: dec(base, 'nvae', mf.sample(b, steps=1))))
    return cfg


def new_stack_configs(dataset, image_size):
    from src.models.ldm_ae import LdmAutoencoder
    from train_ldm_prior import load_latents, build
    import train_ldm_prior as TLP
    Z, _ = load_latents(dataset, image_size)
    C, S, D = TLP.C, TLP.S, TLP.D
    Ztr = Z[:-10000].to(DEVICE)
    mu, sd_ = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True)
    E = Ztr - mu
    Cov = (E.T @ E) / (len(E) - 1)
    L = torch.linalg.cholesky(
        Cov.double() + 1e-4 * torch.eye(D, device=DEVICE, dtype=torch.float64)).float()

    res = '' if image_size == 32 else str(image_size)
    ae = LdmAutoencoder(base=128, z_ch=C, n_res=2).to(DEVICE)
    ae.load_state_dict(torch.load(f'checkpoints/ldmae_{dataset}{res}_z4.pth',
                                  map_location=DEVICE))
    ae.eval()

    cfg = [('per-dim Gaussian',
            lambda b: ae.decode((mu + sd_ * torch.randn(b, D, device=DEVICE)).reshape(-1, C, S, S))),
           ('full-covariance Gaussian',
            lambda b: ae.decode((mu + torch.randn(b, D, device=DEVICE) @ L.T).reshape(-1, C, S, S)))]

    ck = f'checkpoints/ldmae_{dataset}{res}_z4_imm_conv.pth'
    if os.path.exists(ck):
        imm = build('imm', mu.squeeze(0), sd_.squeeze(0), 192, 6)
        imm.load_state_dict(torch.load(ck, map_location=DEVICE)); imm.eval()
        cfg.append(('IMM conv prior (4 NFE)',
                    lambda b: ae.decode(
                        imm.sample(b, imm.sampling_times(4, device=DEVICE)).reshape(-1, C, S, S))))
    return cfg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack', default='new', choices=['old', 'new'])
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--n', type=int, default=N_SAMPLES)
    a = ap.parse_args()

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    cfg = old_stack_configs() if a.stack == 'old' else \
        new_stack_configs(a.dataset, a.image_size)
    out = f'results/{a.dataset}/prdc_{a.stack}{"" if a.image_size == 32 else a.image_size}.csv'
    print(f"\nPRDC on {a.dataset} @ {a.image_size}px, {a.stack} stack, "
          f"{a.n} vs {a.n}, k={K}\n")
    df = evaluate(cfg, fid, a.n, out)
    print(f"\n=== PRECISION / RECALL / DENSITY / COVERAGE ({a.stack} stack) ===")
    print(df[['Prior', 'precision', 'recall', 'density', 'coverage']].to_string(index=False))
