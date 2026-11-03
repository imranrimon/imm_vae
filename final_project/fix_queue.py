"""Sequential GPU queue for the manuscript-correctness fixes.

Two defects need compute, not editing:

(1) Stale IMM ladders. Every IMM row in results/*/ldm_prior.csv predates the
    Variant cache key and was never recomputed after the priors were retrained,
    while the MeanFlow rows on the same plots carry w192b6. The IMM-vs-MeanFlow
    table therefore compares a stale column against a current one.

(2) Cross-crop comparison. The MSE-NVAE stack was trained before the CelebA
    transform was corrected (Resize-then-CenterCrop, which made the crop a
    no-op). Section 2 states we never compare those numbers to the corrected
    ones -- but the diagnostic table places the MSE-NVAE row beside the
    adversarial-AE row and the text reads "the gap grows by a factor of five
    (42 -> 235)" straight across them. Retraining the stack on the corrected
    crop is the only fix that keeps the comparison.

Everything runs sequentially: these all want the whole GPU.

Usage:
    python fix_queue.py [--skip-ladders] [--skip-nvae]
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import subprocess
import sys
import time

PY = sys.executable


def step(title, cmd, tail=2500):
    print(f"\n{'='*70}\n  {title}\n  $ {' '.join(cmd[1:])}\n{'='*70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-tail:])
    if r.returncode != 0:
        sys.stdout.write("\n--- stderr ---\n" + r.stderr[-tail:])
        print(f"  !! FAILED ({time.time()-t0:.0f}s)", flush=True)
        return False
    print(f"  ok ({time.time()-t0:.0f}s)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-ladders', action='store_true')
    ap.add_argument('--skip-nvae', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()

    if not a.skip_ladders:
        step("1/3  Regenerate IMM NFE ladders under an explicit capacity tag",
             [PY, 'regen_imm_ladders.py'], tail=4000)

    if not a.skip_nvae:
        # Stage 1 on the CORRECTED CelebA crop (CenterCrop(148) then resize).
        # src/data.py was fixed earlier, so this simply retrains on correct data.
        step("2/3  Retrain MSE-NVAE Stage 1 on the corrected CelebA crop",
             [PY, 'main.py', '--stage', '1', '--model', 'nvae',
              '--dataset', 'CelebA', '--epochs', str(a.epochs),
              '--method', 'kl', '--run_tag', '_newcrop'], tail=3000)

        step("3/3  Stage-2 priors on the corrected-crop MSE-NVAE latent",
             [PY, 'main.py', '--stage', '2', '--model', 'nvae',
              '--dataset', 'CelebA', '--base_method', 'kl',
              '--method', 'imm', '--run_tag', '_newcrop'], tail=3000)

    print("\nqueue complete", flush=True)


if __name__ == '__main__':
    main()
