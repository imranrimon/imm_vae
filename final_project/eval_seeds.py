"""Seed variance on the headline FIDs.

Every number in the main table is a single run of a stochastic sampler. Two
independent trainings of the CelebA 64x64 prior gave 20.31 and 23.37, so the
spread is not negligible and the paper should quote it. This measures the
*sampling* seed spread with the trained model held fixed, which is the lower
bound on run-to-run variance.

Usage:
    python eval_seeds.py --dataset CelebA --image_size 32 --nfe 4
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import pandas as pd
import torch

from fid_standard import StandardFid
from src.models.ldm_ae import LdmAutoencoder
from train_ldm_prior import load_latents, build
import train_ldm_prior as TLP

DEV = TLP.DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--nfe', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--n', type=int, default=50000)
    a = ap.parse_args()

    Z, _ = load_latents(a.dataset, a.image_size)
    C, S, D = TLP.C, TLP.S, TLP.D
    Ztr = Z[:-10000].to(DEV)
    mu, sd = Ztr.mean(0), Ztr.std(0)

    res = '' if a.image_size == 32 else str(a.image_size)
    ae = LdmAutoencoder(base=128, z_ch=C, n_res=2).to(DEV).eval()
    ae.load_state_dict(torch.load(f'checkpoints/ldmae_{a.dataset}{res}_z4.pth',
                                  map_location=DEV))
    imm = build('imm', mu, sd, 192, 6)
    imm.load_state_dict(torch.load(
        f'checkpoints/ldmae_{a.dataset}{res}_z4_imm_conv.pth', map_location=DEV))
    imm.eval()

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)

    def mk(b):
        return ae.decode(
            imm.sample(b, imm.sampling_times(a.nfe, device=DEV)).reshape(-1, C, S, S))

    vals = []
    for sd_ in a.seeds:
        torch.manual_seed(sd_)
        vals.append(fid.score_stream(mk, a.n))
        print(f"  seed {sd_}: FID {vals[-1]:.3f}", flush=True)

    v = np.array(vals)
    row = dict(Dataset=a.dataset, Res=a.image_size, NFE=a.nfe,
               mean=round(v.mean(), 3), std=round(v.std(ddof=1), 3),
               min=round(v.min(), 3), max=round(v.max(), 3),
               seeds=len(vals))
    out = 'results/seed_variance.csv'
    rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
    rows = [r for r in rows
            if not (r['Dataset'] == a.dataset and r['Res'] == a.image_size)]
    rows.append(row)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{a.dataset}@{a.image_size} NFE={a.nfe}: "
          f"{row['mean']} +/- {row['std']} (n={len(vals)})")


if __name__ == '__main__':
    main()
