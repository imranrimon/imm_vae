"""Over-complete NVAE (s3g2z8, BatchNorm): the numbers Section 6 quotes without a
results row.

The manuscript's NVAE table and its "damping is needed everywhere" paragraph
cite 68.92 / 52.57 (BN recalibrated at t=1.0 / t=0.8), 50.8 (uniform t=0.7) and
53.4-60.8 (coarse groups at t=1, fine groups damped) -- none of which is in any
results file. This recomputes all of them at 50k vs 50k, BN statistics
recalibrated under the exact schedule being evaluated, into
results/CelebA/nvae_proper.csv under variant 'overcomplete-50k'.

Group order is top-down: groups 1-2 at 8x8, 3-4 at 16x16, 5-6 at 32x32.
"Fine groups" = 4-6, matching the mixed-sampling table.

Usage:
    python eval_nvae_overcomplete.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy

import torch

from fid_standard import StandardFid, run
from src.models.nvae_proper import ProperNVAE

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    m = ProperNVAE(base=64, z_ch=8, scales=3, groups_per_scale=2, img=32,
                   norm='batch').to(DEV)
    m.load_state_dict(torch.load('checkpoints/nvaeP_CelebA_s3g2z8.pth',
                                 map_location=DEV))
    m.eval()
    assert m.n_groups == 6, m.n_groups

    schedules = [
        ('BN recal, uniform t=1.0', 1.0),
        ('BN recal, uniform t=0.8', 0.8),
        ('BN recal, uniform t=0.7', 0.7),
        ('BN recal, coarse t=1 / fine (4-6) t=0.6', [1, 1, 1, .6, .6, .6]),
        ('BN recal, coarse t=1 / fine (4-6) t=0.7', [1, 1, 1, .7, .7, .7]),
        ('BN recal, coarse t=1 / fine (4-6) t=0.8', [1, 1, 1, .8, .8, .8]),
    ]
    cfg = []
    for label, t in schedules:
        torch.manual_seed(0)
        mr = copy.deepcopy(m).recalibrate_bn(n=20000, bs=250, device=DEV, t=t)
        cfg.append(('nvae_proper', label, 0,
                    lambda b, mm=mr, x=t: mm.sample(b, DEV, t=x)))

    fid = StandardFid(n_real=50000, dataset='CelebA', image_size=32)
    df = run(cfg, 50000, fid, 'results/CelebA/nvae_proper.csv',
             variant='overcomplete-50k')
    print(df[df.Variant == 'overcomplete-50k'].to_string(index=False))


if __name__ == '__main__':
    main()
