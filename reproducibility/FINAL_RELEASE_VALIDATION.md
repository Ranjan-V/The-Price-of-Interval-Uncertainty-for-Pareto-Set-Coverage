# Final computational release validation

Date: 2026-10-28. Baseline public commit: `5b5283061f7bd506115bca930195ee7d508793ea` on `main`.

A separate clone of the baseline repository received the exact staged Git tree via `git archive`, including Git's LF text rather than relying on the original Windows working files. A new isolated Python environment was created with no system-site packages. It used Windows, Python 3.12.0, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2 and pytest 9.1.1. Direct requirements are in `Code/requirements-final-jgo-revision.txt`; the complete installed package snapshot is `reproducibility/cpu_validation_packages.txt`.

## Executed checks

| Check | Observed result |
| --- | --- |
| Import check | NumPy, SciPy, Matplotlib, interval_pareto and Bank experiment imported successfully |
| `python -m pytest -q Code/tests` | 19 passed, 0 failed |
| CPU smoke | Passed scalar online-bound, front-grid and interval-containment checks; generated trace/summary/plot |
| Essential integrity | 129 hashes verified with LF-normalized text and exact binary hashing |
| Full saved synthetic archive | CRC passed; 2460 successful checkpoints; 246 groups each containing seeds 0–9; 2292 CUDA and 168 CPU |
| All coefficient calculations | 21 cases recomputed and compared to original saved evidence; source hash excluded from comparison because public CLI adaptation changes it |
| Independent SLSQP checks | All 18 cases recomputed; both starts succeeded; maximum relative discrepancy 0.04242618222640715% |
| Final tables | All five CSV tables compared exactly as parsed rows; coefficient and paired-pool CSVs also matched |
| Final computational figures | All three figures regenerated successfully as PDF and PNG; exact PDF byte matching deliberately not required because metadata timestamps vary |
| Bank preprocessing | Every derived NPZ array matched the original saved array exactly |
| Fresh Bank finite-pool run | All 1920 rows and all numerical fields matched saved records; maximum absolute difference 0.0 |
| Synthetic collection | Saved archive extracted; collector rebuilt 2460 per-seed rows and 246 aggregate groups |
| Synthetic plots | Rebuilt uncertainty, regret_vs_U_T, drift, horizon, pareto, joint_heatmap, ablations, regret_vs_bound and front_tracking PNGs |
| Notebook audit | Existing sanitizer/verification compiled public code cells; removed embedded manuscript artifacts: 0 |
| Staged-file review | No manuscript/TeX/private-note paths, credentials matching the checked patterns, personal local paths, or files over 10 MB; `git diff --cached --check` passed |
| Documentation links | All local Markdown links in root README and manifest resolved |

The first new Bank aggregation test assumed zero-based saved pool IDs; it was corrected to read the actual IDs (1–128). The final run passed. The underlying experiment was not changed.

## Exact main validation commands

Run from the root after installing the pinned requirements:

```bash
python Code/validation/verify_final_release.py --run-checks --output-dir release_validation
python Code/experiments/prepare_bank_marketing.py --source Code/data/bank_marketing/bank_marketing_uci.zip --output release_validation/bank_stream.npz
python Code/experiments/real_bank_pareto_coverage.py --data release_validation/bank_stream.npz --output release_validation/bank
python -m zipfile -e reproducibility/result_jcam_synthetic_full.zip release_validation/synthetic_saved
python Code/kaggle/collect_results.py --output-root release_validation/synthetic_saved/results
python Code/kaggle/plot_results.py --output-root release_validation/synthetic_saved/results
python tools/sanitize_notebooks.py
```

The additional Bank comparison loaded every NPZ array with NumPy and required `array_equal`, then compared all saved/recomputed CSV numerical fields with tolerance 1e-12; observed error was zero. Validation outputs/logs remain temporary and are not duplicated in the public source tree. The checked numerical evidence, scripts and recorded outcomes above are public.

## Limits

No large Kaggle GPU training run was repeated. Original saved run completeness, device metadata and collection/plotting were checked instead. Linux/Kaggle instructions were reviewed for portable paths but were not executed on a Linux host during this release. Source redistribution permission was verified on the official UCI page, with attribution retained. The after-query Bank study describes fixed retrospective pools, not cumulative performance in the stationary theorem or future-population coverage. Figure 1 is conceptual manuscript TikZ: its complete public content specification is available, while exact manuscript typography/source is excluded. All computational figures and all table values are reproducible without that source. Numerical diagnostics do not establish theorem proofs or certified global minimax solutions.
