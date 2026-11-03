"""Regenerate the IMM NFE ladders under an explicit capacity tag.

Why this exists: every IMM row in results/*/ldm_prior.csv was written before
fid_standard.run() gained its `Variant` cache key, so those rows carry an empty
variant and were never recomputed after the priors were retrained. The MeanFlow
rows on the same plots DO carry w192b6. Table "IMM vs MeanFlow at matched NFE"
therefore compares a stale IMM column against a current MeanFlow column.

The discrepancy is measurable: eval_seeds.py, run on the current CelebA-32
checkpoint, gives 6.25-6.32 at 4 NFE across three sampling seeds, while the
cached row says 6.492 -- about 5x the seed standard deviation, so the cached row
is not the current model.

Re-running with the explicit tag writes fresh rows alongside the stale ones
(different cache key) so the two can be compared rather than silently replaced.

Usage:
    python regen_imm_ladders.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import sys

import pandas as pd

DATASETS = ['CelebA', 'FFHQ', 'LSUN', 'CIFAR10', 'ImageNet32']


def main():
    for ds in DATASETS:
        print(f"\n{'='*66}\n  {ds}: re-evaluating IMM ladder under tag w192b6\n{'='*66}",
              flush=True)
        r = subprocess.run(
            [sys.executable, 'train_ldm_prior.py', '--dataset', ds,
             '--method', 'imm', '--eval-only', '--fake', '50000'],
            capture_output=True, text=True)
        sys.stdout.write(r.stdout[-2500:])
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-2500:])
            print(f"  !! {ds} FAILED", flush=True)

    print(f"\n{'='*66}\n  STALE vs FRESH\n{'='*66}")
    rows = []
    for ds in DATASETS:
        p = f'results/{ds}/ldm_prior.csv'
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.Prior.str.contains('imm', na=False)]
        old = d[d.Variant.isna() | (d.Variant == '')]
        new = d[d.Variant == 'w192b6']
        for nfe in sorted(set(old.NFE) & set(new.NFE)):
            o = float(old[old.NFE == nfe].FID_std.iloc[0])
            n = float(new[new.NFE == nfe].FID_std.iloc[0])
            rows.append(dict(dataset=ds, NFE=int(nfe), stale=o, fresh=n,
                             delta=round(n - o, 3)))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv('results/imm_ladder_recheck.csv', index=False)
        print(df.to_string(index=False))
        print(f"\nlargest |delta| = {df.delta.abs().max():.3f} FID")
    else:
        print("no overlapping rows found")


if __name__ == '__main__':
    main()
