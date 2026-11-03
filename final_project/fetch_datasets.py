"""Resumable downloads for the two datasets that strengthen the study.

  LSUN church_outdoor  -- a genuinely different domain (scenes). CelebA and FFHQ
                          are both aligned faces, so they are one domain, not two;
                          CIFAR-10 is currently carrying all the diversity.
  ImageNet 32x32       -- the benchmark IMM's own paper reports, so it is the only
                          way to situate our numbers against the method we build on.
                          Official image-net.org copies need credentialed login;
                          this uses an ungated HF mirror instead.

Both are resumable: re-run after an interruption and they continue.

Usage:
    python fetch_datasets.py --which lsun
    python fetch_datasets.py --which imagenet32
"""
import argparse
import os
import sys
import time

import requests

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
LSUN_URL = 'http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip'


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"


def resumable_get(url, dest, chunk=1 << 20):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    head = requests.head(url, timeout=30, allow_redirects=True)
    total = int(head.headers.get('Content-Length', 0))
    if total and have >= total:
        print(f"  already complete: {dest} ({human(have)})")
        return dest
    headers = {'Range': f'bytes={have}-'} if have else {}
    mode = 'ab' if have else 'wb'
    if have:
        print(f"  resuming at {human(have)} / {human(total)}")
    t0, last = time.time(), have
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            for blk in r.iter_content(chunk_size=chunk):
                if not blk:
                    continue
                f.write(blk)
                have += len(blk)
                if have - last > 100 << 20:      # progress every ~100 MB
                    el = time.time() - t0
                    pct = f"{100*have/total:.1f}%" if total else "?"
                    print(f"  {human(have)} / {human(total)}  ({pct})  "
                          f"{(have-(last if el else have))/max(el,1e-9)/1e6:.1f} MB/s",
                          flush=True)
                    last = have
    print(f"  done: {dest} ({human(os.path.getsize(dest))})")
    return dest


def fetch_lsun():
    dest = os.path.join(ROOT, 'LSUN', 'church_outdoor_train_lmdb.zip')
    print(f"LSUN church_outdoor -> {dest}", flush=True)
    resumable_get(LSUN_URL, dest)


def fetch_imagenet32():
    from huggingface_hub import snapshot_download
    out = os.path.join(ROOT, 'ImageNet32')
    print(f"ImageNet 32x32 (HF mirror) -> {out}", flush=True)
    p = snapshot_download(repo_id='ChocolateDave/imagenet-32', repo_type='dataset',
                          local_dir=out, max_workers=4)
    print(f"  done: {p}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', required=True, choices=['lsun', 'imagenet32'])
    a = ap.parse_args()
    (fetch_lsun if a.which == 'lsun' else fetch_imagenet32)()
