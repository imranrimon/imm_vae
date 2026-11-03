"""Resumable 50k FID for the VQ autoregressive prior.

Sampling 50k images from a 64-token autoregressive transformer takes ~950 s,
which exceeds our per-invocation budget. This accumulates Inception features to
disk in chunks and computes FID once enough are banked, so the run can span
several invocations without regenerating anything.

Usage (repeat until it reports the FID):
    python vq_fid_50k.py --chunk 12000
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import linalg

from src.models.vq_ae import VqAutoencoder, CodeTransformer
from fid_standard import StandardFid

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--n_codes', type=int, default=1024)
    ap.add_argument('--quantizer', default='ema')
    ap.add_argument('--tr_dim', type=int, default=512)
    ap.add_argument('--tr_layers', type=int, default=12)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--total', type=int, default=50000)
    ap.add_argument('--chunk', type=int, default=12000)
    ap.add_argument('--bs', type=int, default=500)
    a = ap.parse_args()

    tsz = '' if (a.tr_dim, a.tr_layers) == (256, 6) else f"_d{a.tr_dim}l{a.tr_layers}"
    key = (f"{a.dataset}_c{a.n_codes}_{a.quantizer}{tsz}"
           f"_T{a.temperature}_k{a.top_k}")
    fpath = f"results/{a.dataset}/_vqfeat_{key}.npy"

    feats = np.load(fpath) if os.path.exists(fpath) else np.zeros((0, 2048),
                                                                 dtype=np.float32)
    have = len(feats)
    print(f"banked {have}/{a.total} features for {key}")

    fid = StandardFid(n_real=50000, dataset=a.dataset, image_size=32)

    if have < a.total:
        qsuf = '' if a.quantizer == 'loss' else '_ema'
        ae = VqAutoencoder(base=128, z_ch=4, n_res=2, n_codes=a.n_codes,
                           quantizer=a.quantizer).to(DEVICE)
        ae.load_state_dict(torch.load(
            f"checkpoints/vqae_{a.dataset}_z4_c{a.n_codes}{qsuf}.pth",
            map_location=DEVICE))
        ae.eval()
        tr = CodeTransformer(n_codes=a.n_codes, seq=64, dim=a.tr_dim,
                             heads=8, layers=a.tr_layers).to(DEVICE)
        tr.load_state_dict(torch.load(
            f"checkpoints/vqprior_{a.dataset}_c{a.n_codes}_{a.quantizer}{tsz}.pth",
            map_location=DEVICE))
        tr.eval()

        torch.manual_seed(1000 + have)      # distinct seed per resumed chunk
        want = min(a.chunk, a.total - have)
        new = []
        done = 0
        while done < want:
            b = min(a.bs, want - done)
            with torch.no_grad():
                idx = tr.sample(b, DEVICE, temperature=a.temperature,
                                top_k=a.top_k)
                imgs = ae.decode_codes(idx.view(b, 8, 8))
            new.append(fid.features(imgs.cpu()))
            done += b
            print(f"  {have + done}/{a.total}", flush=True)
        feats = np.concatenate([feats] + new)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        np.save(fpath, feats)
        have = len(feats)

    if have < a.total:
        print(f"banked {have}/{a.total} -- run again to continue")
        return

    mu_r, sg_r = fid.reference()
    f = feats[:a.total]
    mu_f, sg_f = f.mean(0), np.cov(f, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sg_r.dot(sg_f), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    v = float(diff.dot(diff) + np.trace(sg_r) + np.trace(sg_f)
              - 2 * np.trace(covmean))
    print(f"\nFID(50k) = {v:.3f}   [{key}]")

    out = f'results/{a.dataset}/vq.csv'
    rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
    rows.append(dict(Backbone='vq_ae',
                     Prior=f'AR transformer T={a.temperature} top_k={a.top_k or "off"}',
                     NFE=64, Variant=f'c{a.n_codes}-{a.quantizer}{tsz}',
                     FID_std=round(v, 3)))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"appended to {out}")


if __name__ == '__main__':
    main()
