"""Diagnostic row for the MSE-NVAE, recomputed on the CORRECTED CelebA crop.

The paper's diagnostic table places the MSE-NVAE row (Free 42.22, Control 31.07,
Learned 26.37) beside compressors evaluated after the CelebA transform was
fixed, and the surrounding text reads "the gap grows by a factor of five
(42 -> 235)" straight across that boundary. Section 2 says we never make that
comparison, so either the sentence or the row has to change. This recomputes the
row so the sentence can stand.

Timestamps confirm the two regimes are genuinely different data: src/data.py was
corrected at Sep 7 23:12, the image cache (and hence the FID reference) was
rebuilt at 23:16, while the MSE-stack results were written Sep 6 -- before the
fix -- and the adversarial-AE results Sep 8, after it.

Reports the same three quantities as the original row, against the corrected
50k reference:
    Free     z_g ~ N(0, I)
    Control  best temperature over a swept sigma
    Learned  the IMM prior over z_global

Usage:
    python eval_newcrop_diagnostic.py            # after Stage 1 + Stage 2
    python eval_newcrop_diagnostic.py --no-imm   # controls only
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import pandas as pd
import torch

from fid_standard import StandardFid, run, DEVICE, Z_DIM
from src.models.nvae import NVAE

TAG = '_newcrop'


def load_base(tag=TAG):
    ck = f"checkpoints/nvae_CelebA_kl{tag}_stage1.pth"
    assert os.path.exists(ck), f"missing {ck} -- run train_mse_nvae_newcrop.py"
    m = NVAE(z_dim=Z_DIM).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE))
    return m.eval()


def decode(base, z):
    out = base.decoder(z, z_fine=None)
    return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigmas', type=float, nargs='+',
                    default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    ap.add_argument('--fake', type=int, default=50000)
    ap.add_argument('--no-imm', dest='no_imm', action='store_true')
    a = ap.parse_args()

    base = load_base()
    fid = StandardFid(n_real=50000, dataset='CelebA', image_size=32)
    out = 'results/CelebA/newcrop_diagnostic.csv'

    cfg = []
    for s in a.sigmas:
        lbl = ('z_g ~ N(0,I) [free]' if s == 1.0
               else f'z_g ~ N(0,{s}^2 I) [temperature]')
        cfg.append(('nvae_newcrop', lbl, 0,
                    lambda b, s=s: decode(base, s * torch.randn(b, Z_DIM,
                                                                device=DEVICE))))

    if not a.no_imm:
        ck = f'checkpoints/nvae_CelebA_imm{TAG}_stage2.pth'
        if os.path.exists(ck):
            from ab_sampler_eval import imm_sample
            from src.models.imm_core import IMM, OtFmEulerFmParameterization
            from src.models.diffusion import ResidualDenoiser
            net = ResidualDenoiser(input_dim=Z_DIM + 2, model_dim=512,
                                   output_dim=Z_DIM, n_blocks=6)
            m = IMM(net, OtFmEulerFmParameterization(sigma_data=1.0), D=Z_DIM,
                    M=16, convention='legacy').to(DEVICE)
            sd = torch.load(ck, map_location=DEVICE)
            m.load_state_dict(sd, strict=False)
            if 'data_mean' not in sd:
                m.data_mean.zero_(); m.data_std.fill_(1.0)
            m.eval()
            cfg.append(('nvae_newcrop', 'IMM prior over z_global', 16,
                        lambda b: decode(base, imm_sample(m, b, 8, 'legacy'))))
        else:
            print(f"  (no Stage-2 prior at {ck}; controls only)")

    df = run(cfg, a.fake, fid, out, variant='newcrop')
    df = df[df.Variant == 'newcrop']

    free = float(df[df.Prior.str.contains('free')].FID_std.iloc[0])
    temps = df[df.Prior.str.contains('temperature')]
    ctrl = float(temps.FID_std.min()) if len(temps) else float('nan')
    ctrl_s = (temps.loc[temps.FID_std.idxmin(), 'Prior'].split('N(0,')[1].split('^')[0]
              if len(temps) else '-')
    learned = df[df.Prior.str.contains('IMM')]
    learned = float(learned.FID_std.iloc[0]) if len(learned) else None

    print("\n=== MSE-NVAE DIAGNOSTIC ROW, CORRECTED CROP (50k vs 50k) ===")
    print(df.to_string(index=False))
    print(f"\n  Free    (sigma=1.0)      {free:8.2f}     [old crop: 42.22]")
    print(f"  Control (sigma={ctrl_s})       {ctrl:8.2f}     [old crop: 31.07]")
    if learned is not None:
        print(f"  Learned (IMM, 16 NFE)   {learned:8.2f}     [old crop: 26.37]")
        print(f"  Gain  control/learned   {ctrl/learned:8.2f}x    [old crop: 1.18x]")
    print("\nThese are directly comparable to the adversarial-AE row "
          "(234.87 / 30.74 / 6.29).")


if __name__ == '__main__':
    main()
