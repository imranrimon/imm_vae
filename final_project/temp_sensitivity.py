"""Temperature sensitivity across models, at full budget.

Scheduled sampling flattens the NVAE's temperature curve (Section 6). Before
reporting that as evidence about exposure bias, we need to know what a flat or
steep curve actually indicates -- so we sweep temperature on two SINGLE-LEVEL
compressors as a control. They have no context chain and therefore no exposure
bias to speak of, which makes them a falsification test: if steepness tracked
context mismatch, they would be flat.

They are not (see the paper). Temperature steepness is dominated by how sharply
a decoder responds to latent scale, which is a per-model property. What survives
is the within-model comparison: same architecture, same capacity, same data, one
intervention.

Protocol notes:
  * 50k vs 50k standard FID throughout, so these curves are directly comparable
    to every other number in the paper.
  * The NVAE rows recalibrate BatchNorm AT THE TEMPERATURE BEING EVALUATED.
    Recalibrating once at t=1 and then sweeping conflates the two effects; the
    running statistics are a function of the sampling distribution, which the
    temperature changes.
  * Sensitivity is reported as (max - min) / min, which is scale-free. Raw
    spread in FID points is not comparable between models sitting at different
    absolute FIDs.

Usage:
    python temp_sensitivity.py --models nvae,sched,ae,sigma
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import copy

import pandas as pd
import torch

from fid_standard import StandardFid, run

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def nvae_cfgs(dataset, image_size, tag, label, temps, recal):
    from src.models.nvae_proper import ProperNVAE
    m = ProperNVAE(base=64, z_ch=1, scales=3, groups_per_scale=[4, 0, 0],
                   img=image_size, norm='batch').to(DEVICE)
    m.load_state_dict(torch.load(f'checkpoints/{tag}.pth', map_location=DEVICE))
    m.eval()
    cfg = []
    for t in temps:
        if recal:
            mr = copy.deepcopy(m).recalibrate_bn(n=20000, bs=250,
                                                 device=DEVICE, t=t)
            cfg.append(('temp_sens', f'{label} t={t}', 0,
                        lambda b, mm=mr, x=t: mm.sample(b, DEVICE, t=x)))
        else:
            cfg.append(('temp_sens', f'{label} t={t}', 0,
                        lambda b, mm=m, x=t: mm.sample(b, DEVICE, t=x)))
    return cfg


def ae_cfgs(dataset, image_size, sigma_vae, label, temps):
    """Per-dim Gaussian prior on the AE latent, temperature-scaled."""
    from src.models.ldm_ae import LdmAutoencoder
    from train_ldm_prior import load_latents
    import train_ldm_prior as TLP

    Z, _ = load_latents(dataset, image_size, sigma_vae)
    C, S, D = TLP.C, TLP.S, TLP.D
    Ztr = Z[:-10000].to(DEVICE)
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True)

    ck = (f'checkpoints/ldmae_{dataset}_z4_sigma.pth' if sigma_vae
          else f'checkpoints/ldmae_{dataset}_z4.pth')
    ae = LdmAutoencoder(base=128, z_ch=C, n_res=2).to(DEVICE)
    ae.load_state_dict(torch.load(ck, map_location=DEVICE))
    ae.eval()

    return [('temp_sens', f'{label} t={t}', 0,
             lambda b, s=t: ae.decode(
                 (mu + s * sd * torch.randn(b, D, device=DEVICE))
                 .reshape(-1, C, S, S)))
            for t in temps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--fake', type=int, default=50000)
    ap.add_argument('--models', default='nvae,sched,ae,sigma')
    a = ap.parse_args()

    want = set(a.models.split(','))
    temps_h = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # hierarchical: damping regime
    temps_s = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # single-level
    out = f'results/{a.dataset}/temp_sensitivity.csv'
    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)

    specs = []
    if 'nvae' in want:
        specs.append(('NVAE ancestral',
                      lambda: nvae_cfgs(a.dataset, a.image_size,
                                        f'nvaeP_{a.dataset}_s3g400z1',
                                        'NVAE ancestral', temps_h, True)))
    if 'sched' in want:
        specs.append(('NVAE + scheduled sampling',
                      lambda: nvae_cfgs(a.dataset, a.image_size,
                                        f'nvaeP_{a.dataset}_s3g400z1_sched0.3',
                                        'NVAE + sched p=0.3', temps_h, True)))
    if 'ae' in want:
        specs.append(('adversarial AE',
                      lambda: ae_cfgs(a.dataset, a.image_size, False,
                                      'adversarial AE', temps_s)))
    if 'sigma' in want:
        specs.append(('sigma-VAE',
                      lambda: ae_cfgs(a.dataset, a.image_size, True,
                                      'sigma-VAE', temps_s)))

    for label, build in specs:
        print(f"\n--- {label} ---", flush=True)
        run(build(), a.fake, fid, out, variant=f'{a.image_size}px')

    df = pd.read_csv(out)
    df = df[df.Variant == f'{a.image_size}px']
    print(f"\n=== TEMPERATURE SENSITIVITY ({a.dataset} @ {a.image_size}px, "
          f"{a.fake} samples) ===")
    for label, _ in specs:
        sub = df[df.Prior.str.startswith(label.split(' + ')[0])
                 if label != 'NVAE + scheduled sampling'
                 else df.Prior.str.startswith('NVAE + sched')]
        sub = sub[~sub.Prior.str.startswith('NVAE + sched')] \
            if label == 'NVAE ancestral' else sub
        if not len(sub):
            continue
        lo, hi = sub.FID_std.min(), sub.FID_std.max()
        best_t = sub.loc[sub.FID_std.idxmin(), 'Prior'].split('t=')[1]
        print(f"{label:28s} min {lo:7.2f} (t={best_t})  max {hi:7.2f}  "
              f"spread {hi-lo:6.1f}  relative {(hi-lo)/lo:.2f}")


if __name__ == '__main__':
    main()
