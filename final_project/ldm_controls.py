"""Closed-form control priors on an LDM-AE latent.

The headline claim is that a learned prior beats what you get for free. These are
the "for free" baselines, and they must be computed on the SAME latent as the
learned prior, or the comparison is not a control:

  per-dim Gaussian, temperature-swept   -- catches gains that are just rescaling
  full-covariance Gaussian              -- catches gains that are just correlations
  full-covariance + copula              -- adds exact per-dim marginals

On the original NVAE stack these controls beat every learned prior. That is why
they belong in the main body, not the appendix.

Usage:
    python ldm_controls.py --dataset CelebA --image_size 32
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import torch

from src.models.ldm_ae import LdmAutoencoder
from fid_standard import StandardFid, run
from train_ldm_prior import load_latents
import train_ldm_prior as TLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--sigma_vae', action='store_true')
    ap.add_argument('--fake', type=int, default=50000)
    a = ap.parse_args()

    Z, _ = load_latents(a.dataset, a.image_size, a.sigma_vae)
    C, S, D = TLP.C, TLP.S, TLP.D
    dev = TLP.DEVICE
    Ztr = Z[:-10000].to(dev)
    mu = Ztr.mean(0, keepdim=True)
    sd = Ztr.std(0, keepdim=True)

    E = Ztr - mu
    Cov = (E.T @ E) / (len(E) - 1)
    I = torch.eye(D, device=dev, dtype=torch.float64)
    L = torch.linalg.cholesky(Cov.double() + 1e-4 * I).float()
    W = torch.linalg.inv(L.double()).float()

    # per-dim empirical quantiles in whitened space (Gaussian copula)
    U = (Ztr - mu) @ W.T
    K = 4096
    qs = torch.linspace(0, 1, K, device=dev)
    Q = torch.quantile(U.T.contiguous().float(), qs, dim=1).T.contiguous()
    nrm = torch.distributions.Normal(0., 1.)

    def copula(b):
        g = torch.randn(b, D, device=dev)
        p = nrm.cdf(g).clamp(1e-6, 1 - 1e-6) * (K - 1)
        lo = p.floor().long(); hi = (lo + 1).clamp(max=K - 1); w = p - lo.float()
        ql = torch.gather(Q, 1, lo.T).T; qh = torch.gather(Q, 1, hi.T).T
        return mu + ((ql * (1 - w) + qh * w) @ L.T)

    ae_ck = (f'checkpoints/ldmae_{a.dataset}_z4.pth' if a.image_size == 32
             else f'checkpoints/ldmae_{a.dataset}{a.image_size}_z4.pth')
    if a.sigma_vae:
        ae_ck = f'checkpoints/ldmae_{a.dataset}_z4_sigma.pth'
    ae = LdmAutoencoder(base=128, z_ch=C, n_res=2).to(dev)
    ae.load_state_dict(torch.load(ae_ck, map_location=dev))
    ae.eval()

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=a.image_size)
    tag = f'ldm_ae_{a.dataset}{"" if a.image_size == 32 else a.image_size}'
    cfg = []
    for s in [0.6, 0.8, 1.0]:
        cfg.append((tag, f'per-dim Gaussian sigma={s}', 0,
                    lambda b, s=s: ae.decode(
                        (mu + s * sd * torch.randn(b, D, device=dev)).reshape(-1, C, S, S))))
    cfg.append((tag, 'FULL-COV Gaussian', 0,
                lambda b: ae.decode(
                    (mu + torch.randn(b, D, device=dev) @ L.T).reshape(-1, C, S, S))))
    cfg.append((tag, 'FULL-COV + COPULA', 0,
                lambda b: ae.decode(copula(b).reshape(-1, C, S, S))))

    out = (f'results/{a.dataset}/ldm_controls'
           f'{"" if a.image_size == 32 else a.image_size}.csv')
    print(f"\nclosed-form controls on {a.dataset} @ {a.image_size}px "
          f"(latent {C}x{S}x{S} = {D})\n")
    df = run(cfg, a.fake, fid, out,
             variant='closed-form' + ('-sigma' if a.sigma_vae else ''))
    print(f"\n=== CLOSED-FORM CONTROLS: {a.dataset} @ {a.image_size} ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
