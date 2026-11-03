"""Wall-clock sampling cost, to sit beside NFE.

NFE counts network evaluations but not their cost, and the priors compared here
do not share an architecture: the latent DDPM is a 6-block MLP over a 64-d
vector, the conv priors are 4.64M-parameter residual CNNs over a latent grid.
"4 NFE versus 1000 steps" is therefore not a like-for-like efficiency claim
until it is timed.

Reports milliseconds per 1000 generated images, split into latent sampling and
decoding, on the same GPU with CUDA synchronisation and warm-up.

Usage:
    python bench_walltime.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import pandas as pd
import torch

from train_ldm_prior import load_latents, build
import train_ldm_prior as TLP
from src.models.ldm_ae import LdmAutoencoder

DEV = TLP.DEVICE
BATCH = 250
REPS = 4


def timed(fn, warmup=1, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    rows = []

    # ---------------- new stack: conv IMM prior over an 8x8x4 latent ----------
    Z, _ = load_latents('CelebA', 32)
    C, S, D = TLP.C, TLP.S, TLP.D
    Ztr = Z[:-10000].to(DEV)
    mu, sd = Ztr.mean(0), Ztr.std(0)
    ae = LdmAutoencoder(base=128, z_ch=C, n_res=2).to(DEV).eval()
    ae.load_state_dict(torch.load('checkpoints/ldmae_CelebA_z4.pth', map_location=DEV))
    imm = build('imm', mu, sd, 192, 6)
    imm.load_state_dict(torch.load('checkpoints/ldmae_CelebA_z4_imm_conv.pth',
                                   map_location=DEV))
    imm.eval()

    with torch.no_grad():
        dec_t = timed(lambda: ae.decode(torch.randn(BATCH, C, S, S, device=DEV)))
        for nfe in [1, 2, 4, 8, 16]:
            t = timed(lambda n=nfe: imm.sample(BATCH, imm.sampling_times(n, device=DEV)))
            rows.append(dict(Stack='adversarial AE', Prior='IMM conv', NFE=nfe,
                             sample_ms=round(t * 1e3 * 1000 / BATCH, 1),
                             decode_ms=round(dec_t * 1e3 * 1000 / BATCH, 1),
                             total_ms=round((t + dec_t) * 1e3 * 1000 / BATCH, 1)))
        # closed-form controls: a matmul, effectively free
        E = Ztr - Ztr.mean(0, keepdim=True)
        Cov = (E.T @ E) / (len(E) - 1)
        L = torch.linalg.cholesky(
            Cov.double() + 1e-4 * torch.eye(D, device=DEV, dtype=torch.float64)).float()
        t = timed(lambda: torch.randn(BATCH, D, device=DEV) @ L.T)
        rows.append(dict(Stack='adversarial AE', Prior='full-cov Gaussian', NFE=0,
                         sample_ms=round(t * 1e3 * 1000 / BATCH, 1),
                         decode_ms=round(dec_t * 1e3 * 1000 / BATCH, 1),
                         total_ms=round((t + dec_t) * 1e3 * 1000 / BATCH, 1)))

    # ---------------- old stack: MLP priors over a 64-d vector ---------------
    from ab_sampler_eval import load_base, imm_sample, decode as dec
    from src.models.imm_core import IMM, OtFmEulerFmParameterization
    from src.models.diffusion import ResidualDenoiser, DDPM
    ZD = 64
    base = load_base('nvae', 'kl')
    with torch.no_grad():
        dec_o = timed(lambda: dec(base, 'nvae', torch.randn(BATCH, ZD, device=DEV)))

        ddpm = DDPM(ResidualDenoiser(input_dim=ZD + 64, model_dim=512,
                                     output_dim=ZD, n_blocks=6),
                    z_dim=ZD, n_steps=1000, device=DEV).to(DEV)
        ddpm.load_state_dict(torch.load('checkpoints/nvae_CelebA_ddpm_stage2.pth',
                                        map_location=DEV))
        ddpm.eval()
        for st in [100, 1000]:
            t = timed(lambda n=st: ddpm.sample(BATCH, steps=n), reps=2)
            rows.append(dict(Stack='MSE NVAE', Prior='latent DDPM (MLP)', NFE=st,
                             sample_ms=round(t * 1e3 * 1000 / BATCH, 1),
                             decode_ms=round(dec_o * 1e3 * 1000 / BATCH, 1),
                             total_ms=round((t + dec_o) * 1e3 * 1000 / BATCH, 1)))

        net = ResidualDenoiser(input_dim=ZD + 2, model_dim=512, output_dim=ZD, n_blocks=6)
        legacy = IMM(net, OtFmEulerFmParameterization(1.0), D=ZD, M=16,
                     convention='legacy').to(DEV)
        sdict = torch.load('checkpoints/nvae_CelebA_imm_stage2.pth', map_location=DEV)
        legacy.load_state_dict(sdict, strict=False)
        if 'data_mean' not in sdict:
            legacy.data_mean.zero_(); legacy.data_std.fill_(1.0)
        legacy.eval()
        t = timed(lambda: imm_sample(legacy, BATCH, 8, 'legacy'))
        rows.append(dict(Stack='MSE NVAE', Prior='IMM-legacy (MLP)', NFE=16,
                         sample_ms=round(t * 1e3 * 1000 / BATCH, 1),
                         decode_ms=round(dec_o * 1e3 * 1000 / BATCH, 1),
                         total_ms=round((t + dec_o) * 1e3 * 1000 / BATCH, 1)))

    df = pd.DataFrame(rows)
    os.makedirs('results/CelebA', exist_ok=True)
    df.to_csv('results/CelebA/walltime.csv', index=False)
    print("\n=== WALL-CLOCK, ms per 1000 images (RTX 5070 Ti, batch 250) ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
