"""Cross-check numbers asserted in main.tex against the results CSVs.

Point of this file: a claim in the manuscript should be traceable to a run. The
paper has already been wrong twice from numbers that were internally plausible
but not tied to a current result (a stale FID cache, and a table built on a
superseded checkpoint), so the tie is checked mechanically rather than by
reading.

Usage:
    python check_numbers.py
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'final_project', 'results')
TEX = os.path.join(HERE, 'main.tex')
TOL = 0.02

fails, checks = [], 0


def csv(ds, name):
    p = os.path.join(RES, ds, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def pick(df, **kw):
    if df is None:
        return None
    d = df
    for k, v in kw.items():
        if k == 'prior_has':
            d = d[d.Prior.astype(str).str.contains(v, na=False, regex=False)]
        elif k == 'variant':
            d = d[d.Variant.astype(str) == v] if v else d[d.Variant.isna() |
                                                          (d.Variant.astype(str) == 'nan')]
        else:
            d = d[d[k] == v]
    return None if not len(d) else float(d.FID_std.iloc[0])


def check(label, claimed, actual):
    global checks
    checks += 1
    if actual is None:
        fails.append(f"{label}: no matching CSV row (paper says {claimed})")
    elif abs(actual - claimed) > TOL:
        fails.append(f"{label}: paper {claimed}, CSV {actual:.3f}")


def checks_add(label, ok, why):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{label}: {why}")


def main():
    tex = open(TEX, encoding='utf-8').read()

    # ---- main table: learned-prior FIDs come from seed_variance --------------
    sv = os.path.join(RES, 'seed_variance.csv')
    if os.path.exists(sv):
        s = pd.read_csv(sv)
        want = {('CelebA', 32): 6.29, ('FFHQ', 32): 15.16, ('CelebA', 64): 23.35,
                ('LSUN', 32): 22.79, ('CIFAR10', 32): 45.95,
                ('ImageNet32', 32): 75.74}
        for (ds, res), claimed in want.items():
            r = s[(s.Dataset == ds) & (s.Res == res)]
            check(f"tab:main {ds}@{res} learned", claimed,
                  float(r['mean'].iloc[0]) if len(r) else None)

    # ---- NFE table: every cell must be a tagged row -------------------------
    m = re.search(r'\\label\{tab:nfe\}.*?\\end\{tabular\}', tex, re.S)
    if m:
        for ds, key in [('CelebA', 'CelebA'), ('FFHQ', 'FFHQ'),
                        ('LSUN', 'LSUN'), ('CIFAR-10', 'CIFAR10')]:
            blk = re.search(re.escape(ds) + r'\}.*?\\\\', m.group(0), re.S)
            if not blk:
                continue
            vals = [float(x) for x in re.findall(r'(\d+\.\d+)', blk.group(0))]
            d = csv(key, 'ldm_prior.csv')
            for nfe, v in zip([1, 2, 4, 8], vals):
                check(f"tab:nfe {ds} IMM {nfe}NFE", v,
                      pick(d, prior_has='imm conv', NFE=nfe, variant='w192b6'))

    # ---- temperature table --------------------------------------------------
    d = csv('CelebA', 'temp_sensitivity.csv')
    if d is not None:
        for label, model in [('NVAE ancestral', 'NVAE ancestral'),
                             ('scheduled sampling', 'NVAE + sched p=0.3'),
                             ('Adversarial AE', 'adversarial AE'),
                             ('sigma-VAE', 'sigma-VAE')]:
            row = re.search(re.escape(label) + r'[^\\\\]*\\\\', tex)
            if not row:
                continue
            vals = re.findall(r'(\d+\.\d+)', row.group(0))
            if len(vals) < 3:
                continue
            sub = d[(d.Variant == '32px') &
                    d.Prior.str.startswith(model, na=False)]
            if len(sub):
                check(f"tab:temp {label} min", min(float(v) for v in vals[:-2]),
                      round(float(sub.FID_std.min()), 1))

    # ---- NVAE table: every FID in it must exist in a result file -----------
    pool = set()
    for name in ['nvae_proper.csv', 'temp_sensitivity.csv']:
        d = csv('CelebA', name)
        if d is not None:
            pool |= {round(float(v), 2) for v in d.FID_std}
    m = re.search(r'\\label\{tab:nvae\}.*?\\end\{tabular\}', tex, re.S)
    if m:
        body = m.group(0).split('\\midrule', 1)[1]
        for line in body.split('\\\\'):
            if 'Gap' in line or 'dims' in line:
                continue          # ratios / dimension counts, not FIDs
            line = re.sub(r'\(\$t\{=\}[\d.]+\$\)', '', line)   # drop "(t=0.6)"
            for v in re.findall(r'(?<![\d.])(\d+\.\d\d)(?![\d])', line):
                checks_add(f"tab:nvae value {v}", float(v) in pool,
                           "not in nvae_proper.csv or temp_sensitivity.csv")

    # ---- Table 3 rows sourced outside the main-table CSVs -----------------
    nc = csv('CelebA', 'newcrop_diagnostic.csv')
    if nc is not None:
        nc = nc[nc.Variant == 'newcrop']
        want = [round(float(nc[nc.Prior.str.contains('free')].FID_std.iloc[0]), 2),
                round(float(nc[nc.Prior.str.contains('temperature')].FID_std.min()), 2),
                round(float(nc[nc.Prior.str.contains('IMM')].FID_std.iloc[0]), 2)]
        r = re.search(r'MSE NVAE \(\$\\zg\$\)\$\^\\dagger\$ &([^\\]*)', tex)
        got = [float(x) for x in re.findall(r'\d+\.\d+', r.group(1))][:3] if r else []
        checks_add("tab:diagnostic MSE-NVAE free/control/learned", got == want,
                   f"paper {got}, CSV {want}")
    ts = csv('CelebA', 'temp_sensitivity.csv')
    if ts is not None:
        best = round(float(ts[(ts.Variant == '32px') & ts.Prior.str.startswith(
            'NVAE ancestral')].FID_std.min()), 2)
        r = re.search(r'Proper NVAE \(matched\)\s*&\s*[\d.]+\s*&\s*([\d.]+)', tex)
        checks_add("tab:diagnostic proper-NVAE control = temperature-swept best",
                   bool(r) and float(r.group(1)) == best,
                   f"paper {r.group(1) if r else None}, CSV best {best}")

    # ---- values known to be stale: must not reappear -----------------------
    # Each was in the paper at some point and was wrong; the pattern is the
    # exact form it took, so reintroducing it from an old patch fails here.
    stale = {r'FID \$6\.5\$': 'Fig. 2 learned-prior FID (stale cache; 6.29)',
             r'\$45\.79\$': 'CIFAR-10 FID (stale cache; 45.95)',
             r'74\.91|68\.92|52\.57|144\.87': 'over-complete NVAE (no result row)',
             r'\$50\.8\$|53\.4\}?\$?--\$?60\.8': 'damping range (no result row)',
             r'\$0\.3\\times\$\s+worse': 'inverted covariance ratio (3.3x)',
             r'per-dimension std \$0\.98\$': 'abstract statistic (old crop)',
             r'within \$29\.6\$--\$31\.1\$': 'control range incl. MSE-NVAE'}
    for pat, what in stale.items():
        checks_add(f"stale value absent: {what}", not re.search(pat, tex),
                   "reappeared")

    print(f"{checks} numeric claims checked against CSVs")
    if fails:
        print(f"\n{len(fails)} mismatch(es):\n")
        for f in fails:
            print("  ", f)
        sys.exit(1)
    print("all traced to a result row")


if __name__ == '__main__':
    main()
