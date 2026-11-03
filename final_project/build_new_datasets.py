"""Build 32x32 uint8 caches for LSUN church and ImageNet-32.

Both arrive in formats the existing loader doesn't speak (LMDB, parquet), so
they get decoded once into the same `data/_<name>32_<split>.pt` uint8 tensors
every other dataset in this project uses. After this, the whole pipeline
(train_ldm_ae -> train_ldm_prior -> ldm_controls -> make_figures) runs on them
unchanged.

  LSUN church_outdoor -- a non-face, non-object domain. CelebA and FFHQ are both
                         aligned faces, so without this CIFAR-10 carries all the
                         diversity in the study.
  ImageNet-32         -- the benchmark IMM reports, so it is what lets our
                         numbers be situated against the method we build on.

Train sizes are capped near CelebA's 163k so no dataset dominates by scale.

Usage:
    python build_new_datasets.py --which imagenet32
    python build_new_datasets.py --which lsun
"""
import argparse
import io
import os
import zipfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
N_TRAIN = 160000
N_TEST = 10000


def to_u8(img, size=32):
    img = img.convert('RGB')
    if img.size != (size, size):
        # centre-crop to square first so we never squash the aspect ratio
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)


def save(name, split, tensors):
    t = torch.stack(tensors)
    p = os.path.join(ROOT, f'_{name}32_{split}.pt')
    torch.save(t, p)
    print(f"  cached {tuple(t.shape)} -> {p}", flush=True)


def build_imagenet32():
    import pyarrow.parquet as pq
    d = os.path.join(ROOT, 'ImageNet32', 'data')
    shards = sorted(f for f in os.listdir(d) if f.startswith('train-'))
    out = []
    for sh in shards:
        pf = pq.ParquetFile(os.path.join(d, sh))
        for rg in range(pf.num_row_groups):
            for rec in pf.read_row_group(rg, columns=['image']).to_pylist():
                out.append(to_u8(Image.open(io.BytesIO(rec['image']['bytes']))))
                if len(out) >= N_TRAIN:
                    break
            if len(out) >= N_TRAIN:
                break
        print(f"  {sh}: {len(out)} images", flush=True)
        if len(out) >= N_TRAIN:
            break
    save('ImageNet32', 'train', out)

    val = os.path.join(d, 'val-00000-of-00001.parquet')
    pf = pq.ParquetFile(val)
    vo = []
    for rg in range(pf.num_row_groups):
        for rec in pf.read_row_group(rg, columns=['image']).to_pylist():
            vo.append(to_u8(Image.open(io.BytesIO(rec['image']['bytes']))))
            if len(vo) >= N_TEST:
                break
        if len(vo) >= N_TEST:
            break
    save('ImageNet32', 'test', vo)


def build_lsun():
    import lmdb
    zp = os.path.join(ROOT, 'LSUN', 'church_outdoor_train_lmdb.zip')
    ld = os.path.join(ROOT, 'LSUN', 'church_outdoor_train_lmdb')
    if not os.path.exists(ld):
        print(f"  extracting {zp} ...", flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(ROOT, 'LSUN'))
    env = lmdb.open(ld, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = txn.stat()['entries']
        print(f"  LMDB holds {n} images", flush=True)
        out = []
        cur = txn.cursor()
        for i, (_, val) in enumerate(cur):
            out.append(to_u8(Image.open(io.BytesIO(val))))
            if len(out) % 20000 == 0:
                print(f"    {len(out)}", flush=True)
            if len(out) >= N_TRAIN + N_TEST:
                break
    save('LSUN', 'train', out[:-N_TEST])
    save('LSUN', 'test', out[-N_TEST:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', required=True, choices=['lsun', 'imagenet32'])
    a = ap.parse_args()
    print(f"building {a.which} cache", flush=True)
    (build_lsun if a.which == 'lsun' else build_imagenet32)()
