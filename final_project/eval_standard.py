"""Re-score the whole story under the standard FID protocol (canonical
Inception weights, 50k vs 50k), so the numbers are comparable to published work.

Covers both backbones:
  nvae        -- the original over-complete compressor the paper used
  nvae_fine8  -- the compressed compressor (z_fine 8x8x8 = 512 dims)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import torch

from fid_standard import StandardFid, run, DEVICE, Z_DIM
from ab_sampler_eval import load_base, imm_sample, decode as decode_flat
from src.models.imm_core import (IMM, OtFmEulerFmParameterization,
                                 OtFmEulerFmParameterizationFixed)
from src.models.diffusion import ResidualDenoiser, DDPM
from src.models.meanflow import MeanFlow, MeanFlowNet


def orig_configs():
    """The paper's backbone: priors over z_global, z_fine from the decoder."""
    base = load_base('nvae', 'kl')
    base_mmd = load_base('nvae', 'mmd')
    cfg = [('nvae', 'KL-Base  z_g~N(0,1)', 0,
            lambda b: decode_flat(base, 'nvae', torch.randn(b, Z_DIM, device=DEVICE))),
           ('nvae', 'MMD-Base z_g~N(0,1)', 0,
            lambda b: decode_flat(base_mmd, 'nvae', torch.randn(b, Z_DIM, device=DEVICE))),
           ('nvae', 'KL-Base  z_g~N(0,1.5) [temperature]', 0,
            lambda b: decode_flat(base, 'nvae', 1.5 * torch.randn(b, Z_DIM, device=DEVICE)))]

    ddpm = DDPM(ResidualDenoiser(input_dim=Z_DIM + 64, model_dim=512,
                                 output_dim=Z_DIM, n_blocks=6),
                z_dim=Z_DIM, n_steps=1000, device=DEVICE).to(DEVICE)
    ddpm.load_state_dict(torch.load('checkpoints/nvae_CelebA_ddpm_stage2.pth',
                                    map_location=DEVICE))
    ddpm.eval()
    cfg.append(('nvae', 'DDPM prior (paper)', 1000,
                lambda b: decode_flat(base, 'nvae', ddpm.sample(b, steps=1000))))

    net = ResidualDenoiser(input_dim=Z_DIM + 2, model_dim=512, output_dim=Z_DIM, n_blocks=6)
    legacy = IMM(net, OtFmEulerFmParameterization(sigma_data=1.0), D=Z_DIM, M=16,
                 convention='legacy').to(DEVICE)
    sd = torch.load('checkpoints/nvae_CelebA_imm_stage2.pth', map_location=DEVICE)
    legacy.load_state_dict(sd, strict=False)
    if 'data_mean' not in sd:
        legacy.data_mean.zero_(); legacy.data_std.fill_(1.0)
    legacy.eval()
    cfg.append(('nvae', 'IMM-legacy (paper headline)', 16,
                lambda b: decode_flat(base, 'nvae', imm_sample(legacy, b, 8, 'legacy'))))

    p2 = f'checkpoints/nvae_CelebA_imm_fixed_stage2.pth'
    if os.path.exists(p2):
        n2 = ResidualDenoiser(input_dim=Z_DIM + 2, model_dim=512, output_dim=Z_DIM, n_blocks=6)
        fx = IMM(n2, OtFmEulerFmParameterizationFixed(sigma_data=1.0), D=Z_DIM,
                 M=16, convention='fixed').to(DEVICE)
        fx.load_state_dict(torch.load(p2, map_location=DEVICE), strict=False)
        fx.eval()
        cfg.append(('nvae', 'IMM-fixed (corrected convention)', 1,
                    lambda b: decode_flat(base, 'nvae',
                                          fx.sample(b, fx.sampling_times(1, device=DEVICE)))))

    p3 = 'checkpoints/nvae_CelebA_meanflow_stage2.pth'
    if os.path.exists(p3):
        mf = MeanFlow(MeanFlowNet(z_dim=Z_DIM, model_dim=512, n_blocks=6),
                      D=Z_DIM, variant='base').to(DEVICE)
        mf.load_state_dict(torch.load(p3, map_location=DEVICE)); mf.eval()
        cfg.append(('nvae', 'MeanFlow (1-step)', 1,
                    lambda b: decode_flat(base, 'nvae', mf.sample(b, steps=1))))
    return cfg


def fine8_configs():
    """The compressed backbone, including the full-covariance residual prior."""
    from train_joint_prior import load_base as load_c, decode_joint, Z_G
    from train_residual_fine_prior import decoder_fine_prior
    base = load_c()
    d = torch.load('checkpoints/_resid_fine.pt')
    Etr = d['train'].to(DEVICE)
    E = Etr - Etr.mean(0, keepdim=True)
    C = (E.T @ E) / (len(E) - 1)
    L = torch.linalg.cholesky(
        C.double() + 1e-4 * torch.eye(C.shape[0], device=DEVICE,
                                      dtype=torch.float64)).float()
    mu_e = Etr.mean(0, keepdim=True)
    D = C.shape[0]

    def mk(sg, full):
        def f(b):
            zg = sg * torch.randn(b, Z_G, device=DEVICE)
            mu, s = decoder_fine_prior(base, zg)
            e = (mu_e + torch.randn(b, D, device=DEVICE) @ L.T) if full \
                else torch.randn(b, D, device=DEVICE)
            return decode_joint(base, torch.cat([zg, mu + s * e], dim=1))
        return f

    cfg = []
    for sg in [1.0, 1.4]:
        cfg.append(('nvae_fine8', f'z_g~N(0,{sg}) + decoder diagonal', 0, mk(sg, False)))
        cfg.append(('nvae_fine8', f'z_g~N(0,{sg}) + FULL-COV residual', 0, mk(sg, True)))

    rp = 'checkpoints/nvae_CelebA_fine8_residfine_meanflow.pth'
    if os.path.exists(rp):
        from train_residual_fine_prior import build as build_res
        mf = build_res(Etr.mean(0), Etr.std(0), 128, 6)
        mf.load_state_dict(torch.load(rp, map_location=DEVICE)); mf.eval()

        def mk_nn(b):
            zg = torch.randn(b, Z_G, device=DEVICE)
            mu, s = decoder_fine_prior(base, zg)
            e = mf.sample(0, steps=1, cond=zg)
            return decode_joint(base, torch.cat([zg, mu + s * e], dim=1))
        cfg.append(('nvae_fine8', 'z_g~N(0,1) + residual conv MeanFlow', 1, mk_nn))

    from train_joint_prior import joint_latents
    Zva = joint_latents(base, 'test').to(DEVICE)
    cfg.append(('nvae_fine8', 'TRUE encoder latents [ceiling]', 0,
                lambda b: decode_joint(
                    base, Zva[torch.randint(0, len(Zva), (b,), device=DEVICE)])))
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', nargs='+', default=['nvae', 'nvae_fine8'])
    ap.add_argument('--fake', type=int, default=50000)
    ap.add_argument('--real', type=int, default=50000)
    ap.add_argument('--out', default='results/CelebA/fid_standard.csv')
    a = ap.parse_args()

    fid = StandardFid(n_real=a.real)
    cfg = []
    if 'nvae' in a.backbone:
        cfg += orig_configs()
    if 'nvae_fine8' in a.backbone:
        cfg += fine8_configs()
    print(f"standard FID: {a.fake} generated vs {a.real} real, "
          f"pt_inception-2015-12-05 weights\n")
    df = run(cfg, a.fake, fid, a.out)
    print("\n=== STANDARD-PROTOCOL FID ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {a.out}")
