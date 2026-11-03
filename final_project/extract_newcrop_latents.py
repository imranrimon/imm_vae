"""Extract z_global from the corrected-crop MSE-NVAE, for Figure 1(b).

Figure 1(b) plots the per-dimension standard deviation of q(z) for three
compressors. Its MSE-NVAE row was read from checkpoints/_latents_nvae_CelebA.pt,
which was extracted on 5 Sep -- before the CelebA crop was corrected on 7 Sep --
while the sigma-VAE and adversarial-AE rows come from caches built after. That
is the same cross-preprocessing comparison the diagnostic table had, in a figure
instead of a table.

This writes the corrected-crop equivalent. Runs on CPU-resident data with small
batches so it can share the GPU with a training job.

Usage:
    python extract_newcrop_latents.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import torch

from src.models.nvae import NVAE
from train_stage1_fast import image_cache, to_float

OUT = 'checkpoints/_latents_nvae_CelebA_newcrop.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--bs', type=int, default=250)
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = 'checkpoints/nvae_CelebA_kl_newcrop_stage1.pth'
    assert os.path.exists(ck), f"missing {ck}"

    m = NVAE(z_dim=64).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev))
    m.eval()

    out = {}
    for split in ['train', 'val']:
        X = image_cache('CelebA', 'test' if split == 'val' else 'train', 32)
        n = min(a.n, len(X))
        zs = []
        with torch.no_grad():
            for i in range(0, n, a.bs):
                x = to_float(X[i:i + a.bs]).to(dev)
                mu_g, log_g, *_ = m.encoder(x)
                z = mu_g + torch.randn_like(mu_g) * torch.exp(0.5 * log_g)
                zs.append(z.cpu())
        out[split] = torch.cat(zs)
        z = out[split].float()
        sd = z.std(0)
        print(f"  {split}: {tuple(z.shape)}  global mean {z.mean():+.4f}  "
              f"global std {z.std():.4f}  per-dim std [{sd.min():.3f}, {sd.max():.3f}]",
              flush=True)

    torch.save(out, OUT)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
