# The Price of Interval Uncertainty for Pareto Set Coverage

Final JGO computational release. The study examines interval uncertainty, finite-output worst-target Pareto coverage, and when one interval observation improves coverage. The manuscript and private notes are excluded.

**Start with [REPRODUCIBILITY_MANIFEST.md](REPRODUCIBILITY_MANIFEST.md)** for the mapping of four figures and five tables, configurations, environments, hashes and verification status.

## Install the final CPU environment

From the repository root, Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Code/requirements-final-jgo-revision.txt
```

Linux/Kaggle terminal:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Code/requirements-final-jgo-revision.txt
```

In a Kaggle notebook, change directory to the cloned repository, then use `%pip install -r Code/requirements-final-jgo-revision.txt` and prefix the Python commands below with `!`. Restart the kernel if replacing already imported packages. A CPU session is sufficient. The tested final environment is Python 3.12.0, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, pytest 9.1.1; GPU/PyTorch is unnecessary for the final calculations.

## Verify and regenerate the final artifacts

These commands work from the repository root on Windows and Linux:

```bash
python Code/validation/verify_final_release.py
python -m pytest -q Code/tests
python Code/validation/verify_final_release.py --run-checks --output-dir release_validation
```

The last command checks imports, all tests, a CPU smoke run, all 21 recomputed coefficient cases, all 18 independent small-width optimization problems, numerical equality of regenerated tables, creation of all three computational figures, saved-output hashes, and completeness of the 2,460 original synthetic checkpoints. It writes `release_validation/validation_summary.json`. Regenerated PDFs may differ in metadata timestamps. Figure 1 is a conceptual TikZ diagram composed in the manuscript; [its public content specification](Code/PROTOCOL_FLOW.md) permits independent redrawing without publishing manuscript sources.

Individual commands:

```bash
python Code/experiments/diagnose_general_k_strengthening.py --output-dir reproduced/coefficients
python Code/experiments/final_jgo_revision_analysis.py --output-dir reproduced/presentation
python Code/validation/smoke.py --output reproduced/smoke
```

The presentation command reads the original saved coefficient input, Bank records and profile-feedback CSV; it also reruns the independent optimizer. It does not consume the preceding coefficient command's output. The validation command compares the recomputed coefficients separately. New outputs live in `Code/results/final_jgo_revision/`: five tables in CSV/Markdown, three computational figures in PDF/PNG, 21 coefficients, 18 diagnostic records, descriptive statistics and 384 paired pool-level gains. Complete active-constraint, oracle, Jacobian, dual-weight and local checks are in `Code/results/jgo_strengthening/general_k_decision_diagnostics.json`.

## Rebuild the Bank inputs and experiment

UCI explicitly permits sharing/adaptation with attribution under CC BY 4.0 (official page checked 2026-10-28). Existing legitimate source data remain published. The download below is the official source; its expected SHA-256 is `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`.

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; Path('reproduced').mkdir(exist_ok=True); urlretrieve('https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip','reproduced/bank_marketing_uci.zip')"
python Code/experiments/prepare_bank_marketing.py --source reproduced/bank_marketing_uci.zip --output reproduced/bank_stream.npz
python Code/experiments/real_bank_pareto_coverage.py --data reproduced/bank_stream.npz --raw reproduced/bank_marketing_uci.zip --output reproduced/bank
python Code/validation/audit_bank_oracle.py
```

The last command audits the original saved run using an independent grid evaluator/differential evolution and writes its audit JSON in that result directory. Recompressed NPZ containers may have different timestamps; compare arrays and provenance, not only container hashes. Preprocessing uses 8,237 calibration rows, then 128 pools of 256 rows; the last 183 rows are discarded. Age >=40 filtering is within each pool. Five seeds 0â€“4 and observation fractions 0.50/0.80/0.95 produce 1,920 records. Sampling seed is `1000003*seed + 1009*t + int(fraction*100)`, where t=0,...,127 (saved batch ID=t+1). Reading n of N labels with k positives gives the exact containing interval `[k/N,(k+N-n)/N]`. This is fixed-pool containment, not future-population confidence. Legacy classification sensitivity radii are unrelated to the finite-pool intervals.

## Earlier CPU diagnostics and full synthetic suite

The following earlier diagnostics overwrite their corresponding derived result directories; their original saved evidence remains in Git:

```bash
python Code/experiments/pareto_information_lower_bound.py --output reproduced/amplitude --world-count 1024 --u-count 101 --horizon 128
python Code/experiments/check_fixed_regularity_information.py
python Code/experiments/check_k2_coverage_identity.py
python Code/experiments/check_thm10_fixed_k2.py
python Code/experiments/check_profile_class_feedback.py
```

Regenerate the large suite's tables/figures from the existing full archive without retraining:

```bash
python -m zipfile -e reproducibility/result_jcam_synthetic_full.zip reproduced/synthetic_saved
python Code/kaggle/collect_results.py --output-root reproduced/synthetic_saved/results
python Code/kaggle/plot_results.py --output-root reproduced/synthetic_saved/results
```

For an optional new full run, `release_full.json` retains the original matrix/seeds 0â€“9 and redirects output to `reproduced/synthetic/`. On Kaggle use a T4 GPU session with CUDA PyTorch already available:

```bash
python Code/kaggle/run_all.py --stage B --config Code/kaggle/configs/release_full.json --device auto
python Code/kaggle/run_synthetic.py --config Code/kaggle/configs/release_full.json --seed-mode full --device auto
python Code/kaggle/run_all.py --stage H --config Code/kaggle/configs/release_full.json --device auto
```

Stage B checks CPU/CUDA parity on each device. A CPU-only full rerun can use `--device cpu`, with longer runtime. The original notebook used two T4 workers simultaneously; see the historical Kaggle documentation below. Independent processes may use `--seed-mod 2 --seed-rem 0` and `--seed-mod 2 --seed-rem 1`, assigning each process a different GPU with `CUDA_VISIBLE_DEVICES`. No online trajectory is divided across workers. The original archive records 2,292 CUDA and 168 CPU checkpoints in 246 complete ten-seed groups. Large GPU experiments were not rerun for this release.

## Scientific limits

Coefficient calculation evaluates analytic formulas; local perturbations are diagnostics. The 18 independent SLSQP epigraph problems use a separate root evaluator, two starts, 33 hidden worlds, K=1,2,3, widths 1e-4/1e-5 and seed 20261028. Maximum relative discrepancy is 0.0424262%, below 0.043%. They certify neither a global optimizer nor randomized continuum minimaxity. The `1/K` conclusion concerns the coefficient after the fixed-K small-width limit; it does not establish a uniform finite-width joint-(K,s) rate.

Bank statistics describe the same 128 fixed pools; 1,920 correlated records are not independent population replicates. Numerical experiments do not prove the theorems. Older files retain internal theorem numbering; the manifest maps final artifacts. The final public scripts add output-directory controls and numeric table exports while preserving the scientific computations; their source hashes therefore differ from the original private presentation runners.

## Original public research materials

This repository contains the code, executable notebooks, inputs, and numerical results for the study *The Price of Interval Uncertainty for Pareto Set Coverage*. The manuscript is not stored here.

## Contents

- [`Code/src/`](Code/src/), [`Code/experiments/`](Code/experiments/), and [`Code/kaggle/`](Code/kaggle/): implementation and experiment runners.
- [`Code/tests/`](Code/tests/) and [`Code/validation/`](Code/validation/): automated and independent numerical checks.
- [`notebooks/`](notebooks/): the Kaggle notebooks used for synthetic, fixed-regularity, and Bank Marketing runs.
- [`Code/data/bank_marketing/`](Code/data/bank_marketing/): UCI source archive, derived `.npz` stream, and source/preprocessing metadata.
- [`Code/results/`](Code/results/): numerical outputs and figures. The Bank finite-pool run is in [`real_bank_pareto_coverage/`](Code/results/real_bank_pareto_coverage/).
- [`reproducibility/result_jcam_synthetic_full.zip`](reproducibility/result_jcam_synthetic_full.zip): complete Kaggle synthetic-suite output, including 2,460 per-seed checkpoints, environment metadata, tables, and figures (SHA-256 `d181a5fdc24b2281460fd567d556b05b1231425c5c6c7baae705e6a9fbe12bbd`).
- [`Code/EXP12_BANK_PARETO_COVERAGE_REPORT.md`](Code/EXP12_BANK_PARETO_COVERAGE_REPORT.md): data split, exact interval construction, metric, and limitations.

## Reproduce the Bank finite-pool analysis

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r Code/requirements.txt
python Code/experiments/real_bank_pareto_coverage.py
python Code/validation/audit_bank_oracle.py
python -m pytest Code/tests
```

The Bank source archive is the UCI [Bank Marketing dataset](https://doi.org/10.24432/C5K306), attributed to S. Moro, P. Rita, and P. Cortez under CC BY 4.0. [`Code/data/bank_marketing/README.md`](Code/data/bank_marketing/README.md) and its metadata JSON record the source and SHA-256 digest. The experiment treats each chronological batch as a fixed finite pool; it does not infer future-population coverage.

The synthetic suite is documented in [`Code/kaggle/README.md`](Code/kaggle/README.md). The public notebooks retain the executable experiment code and data payloads used on Kaggle; embedded manuscript source and PDF payloads were removed before publication. Several historical notebook-builder scripts refer to manuscript and proof files outside this repository and are retained as provenance rather than as standalone rebuild commands.
