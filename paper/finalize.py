"""Apply the corrected-crop numbers, validate, refresh data files, rebuild the zip.

Run after eval_newcrop_diagnostic.py has written
results/CelebA/newcrop_diagnostic.csv.

Usage:
    python finalize.py [--skip-patch]
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent / 'final_project'
RES = PROJ / 'results'
ZIP = HERE.parent / 'paper_overleaf.zip'
PATCH = pathlib.Path(
    r'C:\Users\rimon\AppData\Local\Temp\claude\G--imm-vae'
    r'\8a6de230-b692-4ebe-981a-5675c9633c7a\scratchpad\patch_paper25.py')

# results files shipped alongside the manuscript, as (source, name-in-zip)
DATA = [
    (RES / 'CelebA' / 'temp_sensitivity.csv', 'temp_sensitivity.csv'),
    (RES / 'CelebA' / 'newcrop_diagnostic.csv', 'newcrop_diagnostic.csv'),
    (RES / 'CelebA' / 'nvae_proper.csv', 'nvae_proper.csv'),
    (RES / 'CelebA' / 'vq.csv', 'vq.csv'),
    (RES / 'CelebA' / 'recon_metrics.csv', 'recon_metrics.csv'),
    (RES / 'CelebA' / 'walltime.csv', 'walltime.csv'),
    (RES / 'CelebA' / 'prdc_new.csv', 'prdc_new.csv'),
    (RES / 'CelebA' / 'prdc_new64.csv', 'prdc_new64.csv'),
    (RES / 'CelebA' / 'prdc_old.csv', 'prdc_old.csv'),
    (RES / 'imm_ladder_recheck.csv', 'imm_ladder_recheck.csv'),
    (RES / 'seed_variance.csv', 'seed_variance.csv'),
]
for ds in ['CelebA', 'FFHQ', 'LSUN', 'CIFAR10', 'ImageNet32']:
    DATA.append((RES / ds / 'ldm_prior.csv', f'ldm_prior_{ds}.csv'))

SKIP_FROM_ZIP = {'check_tex.py', 'check_numbers.py', 'finalize.py'}


def run(cmd, cwd, label):
    print(f"\n--- {label} ---", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-2000:])
        raise SystemExit(f"FAILED: {label}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-patch', action='store_true')
    a = ap.parse_args()

    csv = RES / 'CelebA' / 'newcrop_diagnostic.csv'
    if not a.skip_patch:
        if not csv.exists():
            raise SystemExit(f"missing {csv} -- run eval_newcrop_diagnostic.py first")
        run([sys.executable, str(PATCH)], HERE, "apply corrected-crop numbers")

    run([sys.executable, 'check_tex.py'], HERE, "structure check")
    run([sys.executable, 'check_numbers.py'], HERE, "numeric check")

    print("\n--- refresh shipped data files ---")
    n = 0
    for src, name in DATA:
        if src.exists():
            shutil.copy2(src, HERE / name)
            n += 1
        else:
            print(f"  (absent, skipped) {src.name}")
    print(f"  {n} results files refreshed")

    print("\n--- rebuild zip ---")
    files = 0
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.rglob('*')):
            if (p.is_dir() or p.name in SKIP_FROM_ZIP
                    or p.suffix in {'.aux', '.log', '.out', '.synctex'}
                    or '__pycache__' in p.parts):
                continue
            z.write(p, p.relative_to(HERE))
            files += 1
    print(f"  {ZIP}: {files} files, {ZIP.stat().st_size/1e6:.2f} MB")
    print("\nfinalize: OK")


if __name__ == '__main__':
    main()
