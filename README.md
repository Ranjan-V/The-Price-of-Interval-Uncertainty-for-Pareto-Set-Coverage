# Interval uncertainty for Pareto set coverage: reproducibility files

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
