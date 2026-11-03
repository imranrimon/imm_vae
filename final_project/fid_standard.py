"""Standard-protocol FID: canonical Inception weights, 50k samples.

Earlier numbers in this project used torchvision's ImageNet InceptionV3 with
5k generated / 10k real. Those are internally consistent but NOT comparable to
published FIDs. This module uses:

  * pytorch_fid's InceptionV3 -- the TF-ported pt_inception-2015-12-05 network
    that the original FID paper and every published number are based on
  * 50,000 generated vs 50,000 real (reference drawn from the CelebA train
    split, the usual convention), instead of 5k/10k

The reference statistics are cached, so only the generated pass costs anything
per configuration.

Usage:
    python fid_standard.py --model nvae            # original (paper) backbone
    python fid_standard.py --model nvae_fine8      # compressed backbone
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from pytorch_fid.inception import InceptionV3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_DIM = 64


class StandardFid:
    def __init__(self, n_real=50000, dataset='CelebA', image_size=32):
        self.n_real = n_real
        self.dataset = dataset
        self.image_size = image_size
        self.model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE).eval()
        self._ref = None

    @torch.no_grad()
    def features(self, imgs_uint8_or_float, bs=100):
        """Accepts images in [-1,1] float; InceptionV3 wants [0,1]."""
        out = []
        for i in range(0, len(imgs_uint8_or_float), bs):
            b = imgs_uint8_or_float[i:i + bs].to(DEVICE)
            if b.dtype == torch.uint8:
                b = b.float().div_(255.0)
            else:
                b = (b * 0.5 + 0.5).clamp(0, 1)
            f = self.model(b)[0].squeeze(-1).squeeze(-1)
            out.append(f.cpu().numpy())
        return np.concatenate(out)

    def reference(self):
        if self._ref is not None:
            return self._ref
        cache = f"results/{self.dataset}/_stdfid_ref_{self.image_size}_{self.n_real}.npz"
        if os.path.exists(cache):
            d = np.load(cache)
            self._ref = (d['mu'], d['sigma'])
            print(f"  reference: {self.n_real} real (cached)")
            return self._ref
        X = torch.load(f"data/_{self.dataset}{self.image_size}_train.pt")
        g = torch.Generator().manual_seed(1234)
        idx = torch.randperm(len(X), generator=g)[:self.n_real]
        print(f"  computing reference over {self.n_real} real images...", flush=True)
        f = self.features(X[idx])
        mu, sigma = f.mean(0), np.cov(f, rowvar=False)
        os.makedirs(f'results/{self.dataset}', exist_ok=True)
        np.savez(cache, mu=mu, sigma=sigma)
        self._ref = (mu, sigma)
        return self._ref

    def score(self, imgs):
        mu_r, sg_r = self.reference()
        f = self.features(imgs)
        mu_f, sg_f = f.mean(0), np.cov(f, rowvar=False)
        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sg_r.dot(sg_f), disp=False)
        if not np.isfinite(covmean).all():
            off = np.eye(sg_r.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sg_r + off).dot(sg_f + off))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sg_r) + np.trace(sg_f)
                     - 2 * np.trace(covmean))

    def score_stream(self, make_batch, n, bs=250):
        """Score without materializing all n images.

        At 64x64, 50k float images is ~2.5 GB and generate() briefly holds two
        copies. Extracting features per batch keeps memory flat.
        """
        mu_r, sg_r = self.reference()
        feats, done = [], 0
        while done < n:
            b = min(bs, n - done)
            with torch.no_grad():
                imgs = make_batch(b)
            feats.append(self.features(imgs.cpu()))
            done += b
            del imgs
        f = np.concatenate(feats)[:n]
        mu_f, sg_f = f.mean(0), np.cov(f, rowvar=False)
        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sg_r.dot(sg_f), disp=False)
        if not np.isfinite(covmean).all():
            off = np.eye(sg_r.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sg_r + off).dot(sg_f + off))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sg_r) + np.trace(sg_f)
                     - 2 * np.trace(covmean))


def generate(n, make_batch, bs=500):
    out, done = [], 0
    while done < n:
        b = min(bs, n - done)
        with torch.no_grad():
            out.append(make_batch(b).cpu())
        done += b
    return torch.cat(out)[:n]



def run(configs, n_fake, fid, out_csv, variant=""):
    """variant distinguishes runs that share a (backbone, prior, NFE) key but not
    a model -- capacity, protocol, crop. Without it, re-running a config with a
    different model silently returns the previous model's cached row; that
    happened twice during this project before the key was widened."""
    rows = []
    if os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict('records')
        for r in rows:
            r.setdefault('Variant', '')
    done = {(r['Backbone'], r['Prior'], r['NFE'], str(r.get('Variant', '')))
            for r in rows}
    for backbone, name, nfe, mk in configs:
        if (backbone, name, nfe, variant) in done:
            print(f"  {backbone:11s} {name:44s} cached")
            continue
        torch.manual_seed(0)
        f = fid.score_stream(mk, n_fake)
        rows.append(dict(Backbone=backbone, Prior=name, NFE=nfe,
                         Variant=variant, FID_std=round(f, 3)))
        done.add((backbone, name, nfe, variant))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  {backbone:11s} {name:44s} NFE={nfe:>4}  FID_std={f:8.3f}", flush=True)
    return pd.DataFrame(rows)
