# Overleaf package — corrected results

Upload `paper_overleaf.zip` to Overleaf (New Project → Upload Project).
Compile `main.tex` with pdfLaTeX. It uses `acmart` (`sigconf,nonacm`), which
Overleaf provides; the bibliography is inline (`thebibliography`), so no
`.bib` run is needed.

## What changed from the previous draft

The framing inverted. The old draft argued that an expressive latent prior
improves generation. The corrected result is that a learned prior helps **in
proportion to the gap between the aggregate posterior q(z) and the prior
p(z)**, and that gap is set by the Stage-1 compressor. The paper now measures
that gap and benchmarks against controls that cost zero network evaluations.

### Numbers that changed and why

| Claim in old draft | Status | Corrected |
|---|---|---|
| FID 33.35, "state of the art" | metric was non-standard | 26.37 under canonical Inception at 50k; a scalar temperature reaches 31.07 |
| IMM beats DDPM | holds only on NVAE | false on VAE (100.8 vs 34.8); DDPM was also denied the normalization IMM received |
| 64×64 throughout | code ran 32×32 | both resolutions now reported, at fixed compression ratio |
| NVAE gives 10× better reconstruction | latent was 2.7× over-complete | at genuine 5.3× compression, MSE₂₅₅ 40.4 vs the flat VAE's 56.1 |
| Table 2 VAE+IMM row | not reproducible | reruns give 120/105/101/101 |

### This revision — audit fixes

Five defects were found by auditing the manuscript against the result files,
and all five needed re-running rather than re-wording.

| Defect | How it surfaced | Fix |
|---|---|---|
| Table 6's IMM column was stale cache | Tables 3 and 6 disagreed about the same model (6.29 vs 6.49) | all five ladders recomputed under an explicit capacity tag; IMM still wins 16/16 |
| Section 3 asserted the opposite of Section 6 | §3 called the matched NVAE ceiling "evidence the gap is not capacity"; §6 shows it *was* capacity (21,504 vs 256 dims) | claim removed, replaced with what the row supports |
| Table 3 compared across two preprocessings | §2 promises this never happens; the MSE-NVAE row was scored before the CelebA crop fix, the rest after | MSE-NVAE stack retrained on the corrected crop; Table 3 is now single-regime |
| "Gain falls monotonically with domain diversity" | FFHQ (aligned faces, 2.4×) ranks below LSUN (scenes, 3.4×) | replaced with what the six settings support |
| "Free = N(0,I)" | it is a per-dimension moment-matched Gaussian, a stronger baseline | caption corrected |

### Second audit (22 fixes)

A full read against the result files found 22 inconsistencies that the earlier
validators could not see. The ones that change something a reader would rely on:

- **Compile bug.** Two header rows of the temperature table ended in a single
  backslash instead of the double backslash that ends a LaTeX table row, so the
  previous zip would not compile on Overleaf.
- **Unsourced numbers.** The over-complete column of the NVAE table and the
  "damping is needed everywhere" paragraph quoted values (68.92, 52.57, 74.91,
  50.8, 53.4-60.8) that existed in no results file. All were re-run at 50k
  (`final_project/eval_nvae_overcomplete.py`); the conclusion holds
  (fine-only damping 53.5-59.4 vs uniform 50.76), the numbers moved.
- **The diagnostic statistic.** The paper ordered compressors by mean
  per-dimension std (1.00 / 1.01 / 1.66), which does not separate the first two.
  Mean per-dimension KL to N(0,1) does (0.025 / 0.048 / 0.664), and Figure 1(c)
  now plots it instead of free FID -- which the text itself says is not the
  diagnostic.
- **VQ "best headroom" was an apples-to-oranges comparison.** 98.7% is measured
  from VQ's free baseline; measured the same way the adversarial AE is 98.2%.
- **Leftovers from the crop retrain** in the abstract, Section 3, Figure 1 and
  the results text, plus two stale cache values (6.5, 45.79) and an inverted
  ratio (0.3x for 3.3x).

`check_tex.py` now rejects rows ending in a single backslash, and
`check_numbers.py` covers 45 claims (from 22), including every FID in the NVAE
table and a list of known-stale values that must not reappear. Run against the
pre-audit manuscript it reports 12 failures.

### The crop retrain, and what it changed

Retraining the MSE-NVAE stack on the corrected CelebA crop moved every absolute
FID on that row by 3-4x, while leaving the ratio the paper's claim is about
essentially unchanged:

| | old crop | corrected crop |
|---|---|---|
| Free (sigma=1.0) | 42.22 | 155.59 |
| Control (best sigma) | 31.07 @ 1.5 | 104.20 @ 2.2 |
| Learned (IMM, 16 NFE) | 26.37 | 90.71 |
| Ceiling (rFID) | not reported | 1.665 |
| **Gain (control/learned)** | **1.18x** | **1.15x** |

Two things this exposed:

- **Free FID is not a ladder.** Across the four compressors it spans only 2.2x,
  and the MSE-NVAE's free baseline is now *worse* than the sigma-VAE's despite
  its aggregate posterior being closer to the prior. The paper's earlier
  "the gap grows by a factor of five (42 -> 235)" does not survive. What does
  survive, and is now the stated claim, is the aggregate-posterior statistic:
  mean per-dimension deviation 1.00 / 1.01 / 1.66 orders the gains 1.15x /
  2.1x / 4.9x.
- **Table 3 had a capacity confound.** The MSE-NVAE row is 8256 latent
  dimensions against 256 for every other row, and its free sampler draws only
  the 64-d z_global while the decoder's conditional prior supplies the other
  8192. It now carries a footnote saying so.

A near-miss worth recording: the first Stage-2 attempt was killed by the OS for
memory pressure partway through, and its checkpoint gave FID 128.53 (gain
0.81x) -- i.e. it said the learned prior *loses* to a scalar temperature. Run to
its full 100 epochs the same configuration gives 90.71 (1.15x). The interrupted
number was self-consistent and would have supported a striking, false headline;
only the budget was wrong.

Two claims of our own were tested and dropped:

- **"Scheduled sampling flattens the temperature curve 4.8×."** An artifact of
  sweeping t ∈ [0.6, 1.0], which straddles the baseline's optimum but clips the
  fine-tuned model's. On the full grid both models have *identical* sensitivity
  (width 0.34). What survives is that the optimum **moves 0.6 → 0.9** and FID
  improves 44.13 → 40.88 — the exposure-bias prediction.
- **"Temperature steepness diagnoses train/test context mismatch."** Falsified
  by a control: two single-level compressors, which have no context chain, have
  *sharper* optima (0.19, 0.22) than either hierarchical model.

The regeneration carries its own control: on LSUN and ImageNet-32 all ten
cached rows reproduced to Δ = 0.000, establishing that sampling and scoring are
deterministic here — which is what makes CelebA's and FFHQ's shifts attributable
to a changed checkpoint rather than sampler noise.

Added: `figures/overview.pdf` (Figure 1), drawn from measured latent statistics
rather than free-hand; `check_tex.py` (structure) and `check_numbers.py`
(22 numeric claims traced to CSV rows), since the paper has now twice been wrong
from numbers that were plausible but not tied to a current run.

### Protocol

All FIDs: canonical TF-ported InceptionV3 (`pytorch-fid` weights), 50,000
generated vs 50,000 training images. One Stage-1 protocol and one Stage-2
architecture across every dataset and resolution, no per-dataset tuning.
CelebA uses `CenterCrop(148)` then resize; FFHQ uses a 60k/10k split.

## Files

```
main.tex            the paper
figures/            sample grids (reconstructions, samples, compare)
all_results.csv     every FID row behind the tables
```

## Still to add before submission

1. **Precision/recall or coverage.** The paper's own caution is that FID
   rewards variance rescaling; FID alone cannot rebut that.
2. **Seed variance** on the headline numbers. Only the MeanFlow NFE sweep has
   been repeated across seeds (3 seeds, spread ~0.3 FID).
3. **Wall-clock** alongside NFE.
4. **CelebA 64×64 sample grids** — reconstructions exist; samples/compare need
   the 64×64 prior re-run under the corrected checkpoint naming.
5. **LSUN and ImageNet-32** are downloaded (`data/LSUN`, `data/ImageNet32`) but
   not yet unpacked into loaders.
