"""Qualitative figures, sized to be legible in print.

The first version tiled 32 images per row at native 32x32. Scaled to a paper
column that is a few millimetres per face -- unreadable. This version:

  * uses 8 columns, so each tile keeps a usable fraction of the column width
  * upscales with NEAREST interpolation (never bilinear): these are 32x32
    images and smoothing them invents detail the model did not produce, which
    is exactly the wrong thing to do in a figure about sample sharpness
  * labels every row, so the reader does not have to consult the caption to
    know which prior produced which block

Outputs per dataset:
  reconstructions_<ds>.png   held-out reals vs their reconstructions
  samples_<ds>.png           prior samples at 1, 2, 4, 8 NFE
  compare_<ds>.png           Gaussian / full-covariance / learned prior

Usage:
    python make_figures.py --dataset CelebA --image_size 32
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.ldm_ae import LdmAutoencoder
from train_stage1_fast import image_cache, to_float
from train_ldm_prior import load_latents, build
import train_ldm_prior as TLP

OUT = 'results/figures'
NCOL = 8
UPSCALE = 4


def _to_np(x):
    """[B,3,H,W] in [-1,1] -> [B,H,W,3] uint8, nearest-upscaled."""
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    return np.repeat(np.repeat(x, UPSCALE, axis=1), UPSCALE, axis=2)


def panel(rows, labels, path, title=None):
    """rows: list of [NCOL,3,H,W] tensors, one per labelled row."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nr = len(rows)
    imgs = [_to_np(r[:NCOL]) for r in rows]
    h, w = imgs[0].shape[1:3]

    fig_w = NCOL * w / 100 * 1.0
    fig_h = nr * h / 100 * 1.0 + 0.35 * nr
    fig, axes = plt.subplots(nr, NCOL, figsize=(fig_w, fig_h),
                             gridspec_kw=dict(wspace=0.03, hspace=0.04))
    axes = np.atleast_2d(axes)
    for i in range(nr):
        for j in range(NCOL):
            ax = axes[i, j]
            ax.imshow(imgs[i][j], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(labels[i], rotation=0, ha='right', va='center',
                              fontsize=9, labelpad=8)
    if title:
        fig.suptitle(title, fontsize=10, y=0.995)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    a = ap.parse_args()
    dev = TLP.DEVICE
    sfx = '' if a.image_size == 32 else str(a.image_size)

    ae = LdmAutoencoder(base=128, z_ch=4, n_res=2).to(dev)
    ae.load_state_dict(torch.load(f'checkpoints/ldmae_{a.dataset}{sfx}_z4.pth',
                                  map_location=dev))
    ae.eval()

    Z, _ = load_latents(a.dataset, a.image_size)
    C, S, D = TLP.C, TLP.S, TLP.D
    Ztr = Z[:-10000].to(dev)
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True)

    # ---- reconstructions (held out) ----
    X = image_cache(a.dataset, 'test', a.image_size)
    torch.manual_seed(0)
    idx = torch.randperm(len(X))[:NCOL]
    real = to_float(X[idx]).to(dev)
    with torch.no_grad():
        rec, _, _, _ = ae(real, sample=False)
    panel([real, rec], ['real', 'recon'],
          f'{OUT}/reconstructions_{a.dataset}{sfx}.png',
          title=f'{a.dataset} {a.image_size}x{a.image_size} — reconstruction')

    # ---- closed-form baselines ----
    E = Ztr - mu
    Cov = (E.T @ E) / (len(E) - 1)
    L = torch.linalg.cholesky(
        Cov.double() + 1e-4 * torch.eye(D, device=dev, dtype=torch.float64)).float()
    torch.manual_seed(1)
    with torch.no_grad():
        g_naive = ae.decode((mu + sd * torch.randn(NCOL, D, device=dev)).reshape(-1, C, S, S))
        g_cov = ae.decode((mu + torch.randn(NCOL, D, device=dev) @ L.T).reshape(-1, C, S, S))

    ck = f'checkpoints/ldmae_{a.dataset}{sfx}_z4_imm_conv.pth'
    imm = build('imm', mu.squeeze(0), sd.squeeze(0), 192, 6)
    imm.load_state_dict(torch.load(ck, map_location=dev)); imm.eval()

    rows, labels = [], []
    for st in [1, 2, 4, 8]:
        torch.manual_seed(2)
        with torch.no_grad():
            z = imm.sample(NCOL, imm.sampling_times(st, device=dev))
            rows.append(ae.decode(z.reshape(-1, C, S, S)))
        labels.append(f'{st} NFE')
    panel(rows, labels, f'{OUT}/samples_{a.dataset}{sfx}.png',
          title=f'{a.dataset} — IMM prior samples by step count')

    torch.manual_seed(2)
    with torch.no_grad():
        z = imm.sample(NCOL, imm.sampling_times(4, device=dev))
        g_learn = ae.decode(z.reshape(-1, C, S, S))
    panel([g_naive, g_cov, g_learn],
          ['Gaussian', 'full-cov', 'IMM (4 NFE)'],
          f'{OUT}/compare_{a.dataset}{sfx}.png',
          title=f'{a.dataset} — same decoder, three priors')


if __name__ == '__main__':
    main()
