# Final JGO computational reproducibility manifest

Release scope: the final 30-page submission, four figures and five tables. Manuscript sources, proofs, author declarations and internal reviews are excluded. Original public history and artifacts are preserved. This manifest maps computation rather than reproducing the manuscript.

## Commands and environment keys

All commands run from the repository root. Install `Code/requirements-final-jgo-revision.txt` as described in README. **E1** = CPU, Python 3.12.0, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2; release tests use pytest 9.1.1. **E2** = original local Bank CPU environment, Python 3.11.9/NumPy 1.26.4/SciPy 1.11.1/Matplotlib 3.10.8. **E3** = original Kaggle synthetic environment, Linux, four logical CPU cores, two Tesla T4s, Python 3.12.13/NumPy 2.0.2/Matplotlib 3.10.0/PyTorch 2.10.0+cu128. **E4** = manuscript TikZ/LaTeX, no numerical computation. Bank Kaggle execution used CPU; its exact package versions were not retained. Hardware/software provenance is machine-readable in `Code/configs/final_jgo_environments.json`.

```bash
# C1: coefficient/oracle/Jacobian/dual-weight and local one-query checks
python Code/experiments/diagnose_general_k_strengthening.py --output-dir reproduced/coefficients
# C2: all computational manuscript figures, tables and independent optimization
python Code/experiments/final_jgo_revision_analysis.py --output-dir reproduced/presentation
# C3: amplitude-scaled diagnostics
python Code/experiments/pareto_information_lower_bound.py --output reproduced/amplitude --world-count 1024 --u-count 101 --horizon 128
# C4: real-data calculation using bundled verified inputs
python Code/experiments/real_bank_pareto_coverage.py --output reproduced/bank
# C5: validate hashes/checkpoint completeness, then rerun essential CPU checks
python Code/validation/verify_final_release.py --run-checks --output-dir release_validation
```

For the table below, **F** means `Code/results/final_jgo_revision/`. CSV/Markdown table exports carry full precision; manuscript formatting rounds those values. Table 5 is provenance rather than experimental output. Figure 1 and Tables 1/3/4/5 were typeset directly in LaTeX; the latter tables' values are independently exported here without manuscript sources. Table 2 was inserted from a generated LaTeX fragment; the public script exports the same values in CSV/Markdown, with no LaTeX dependency.

| Manuscript artifact | Source script / configuration | Input | Saved public output | Regenerate / environment | Verification |
| --- | --- | --- | --- | --- | --- |
| Figure 1: project/protocol flow | Manual TikZ composition; scientific content in `Code/PROTOCOL_FLOW.md` | No data or seed | Content specification; exact manuscript TikZ deliberately excluded | Redraw from content specification / E4 | Conceptual artifact, no numerical check; exact original typography requires manuscript source |
| Figure 2: one-query precision | `Code/experiments/final_jgo_revision_analysis.py`, `profile_plot()`; beta=1, T=128, noise factors 0/.1/.5 | `Code/results/profile_class_feedback/profile_class_feedback.csv` | F`profile_query_precision.pdf`, `.png` | C2 / E1 | Regenerated from saved direct continuous-coverage diagnostics |
| Figure 3: coefficient scaling | Same script, `coefficients()`; profiles A/B/C, seven K values | `Code/results/jgo_strengthening/general_k_decision_diagnostics.json` | F`theorem6_coefficient_scaling.pdf`, `.png` | C1 then C2 / E1 | 21 cases/bounds checked; unfitted 0.25/K reference; no uniform finite-width claim |
| Figure 4: Bank sampling variability | Same script, `bank()`; three observation fractions, five seeds, 128 pools | `Code/results/real_bank_pareto_coverage/bank_pareto_coverage_rounds.csv` | F`bank_pareto_variability.pdf`, `.png` | C2; optionally C4 / E1 (original E2) | Five seed-wise means and their range; no population confidence interval |
| Table 1: scaled quadratics | `pareto_information_lower_bound.py` plus presentation `tables()`; amplitude=1 displayed, all 25 saved configurations retained | `Code/results/thm06_pareto_information/pareto_information_summary.json` | F`table1_scaled_coverage.csv`, `.md` | C3 for original experiment; C2 for table / E1 CPU | Five displayed rows; all 25 corrected checks retained; minimum certified ratio 1.9243149902343744 |
| Table 2: first-order coefficients | `diagnose_general_k_strengthening.py`; presentation `coefficients()`/`tables()` | Full 21-case JSON above | F`theorem6_coefficients.csv`, `table2_coefficients.csv`, `.md` | C1 then C2 / E1 | Formula evaluation, oracle/active/dual residuals; separate independent diagnostic below |
| Table 3: Bank means | `real_bank_pareto_coverage.py`; presentation `bank()`/`tables()` | All 1,920 Bank records | F`table3_bank_main.csv`, `.md`; original summary in Bank result directory | C4 then C2 / E1; original E2 | Means, labels read, adaptive/static ratios and improved-case fractions recomputed |
| Table 4: Bank dispersion/gains | Presentation `bank()`/`tables()` | Same records; pool-averaged five-seed values | F`table4_bank_dispersion.csv`, `.md`, `bank_descriptive_statistics.json`, `bank_pool_seedmeans.csv` | C2 / E1 | 384 pool/fraction rows; paired static minus adaptive; pool SD, within-pool RMS SD, gain SD checked |
| Table 5: computation environments | Presentation `tables()` | `Code/configs/final_jgo_environments.json`; synthetic ZIP logs/checkpoints | F`table5_computing_environments.csv`, `.md` | C2 / E1; reports E1/E2/E3 and older CPU provenance | Hardware author-supplied; recorded package versions distinguished from unavailable notebook versions |

## Theorem 6 coefficient and diagnostic configuration

`R=0.125`, profiles `phi(v)=scale*(v^2+beta*v^4)`:

| Pair | Left (scale,beta) | Right (scale,beta) | K |
| --- | --- | --- | --- |
| A | (1,0) | (1,0) | 1,2,3,4,8,16,32 |
| B | (1,1) | (1,1) | 1,2,3,4,8,16,32 |
| C | (1,0) | (3,2) | 1,2,3,4,8,16,32 |

C1 uses standard-library bisection, greedy oracle feasibility, active endpoint/gap roots, a chain Jacobian and positive left-null weights. It records 120 seeded local perturbations per case, dense targets, small-width diagnostics and six one-query checks. Seed=20261028. These evaluate formulas and probe inequalities; they do not independently optimize the continuum minimax problem.

C2's **independent** diagnostic reimplements continuous-target roots with SciPy Brent and minimizes a deterministic epigraph by SLSQP. For each of three profile pairs, K=1,2,3 and s=1e-4/1e-5 give **18** cases. Each uses 33 hidden worlds and two starts (zero and a fresh uniform [-.2,.2] random perturbation), RNG seed=20261028, 160 iterations, `ftol=1e-9`, root `xtol=5e-15`, scaled offsets bounded [-3,3]. Both starts succeeded in every saved case. The saved F`theorem6_independent_smallwidth.json` yields maximum absolute relative discrepancy `0.0004242618222640715`, or **0.04242618222640715%**, below the reported 0.043%. This is neither certified global optimization nor randomized continuum-world minimax verification. Fixed-K asymptotics do not imply a uniform joint-(K,s) finite-width rate.

Original private source hashes are retained in the saved coefficient provenance (`script_sha256`) and were compared before packaging. The public scripts add output-directory controls; the presentation version also replaces private manuscript writes with public CSV/Markdown table exports. These interface changes explain changed source hashes. Numerical computations and saved raw inputs are preserved. `presentation_audit.json` records LF-normalized public presentation script and input hashes.

## Earlier numerical claims and full synthetic provenance

| Evidence | Runner | Saved inputs/outputs | Configuration and status |
| --- | --- | --- | --- |
| Amplitude quadratics | `Code/experiments/pareto_information_lower_bound.py` | `Code/results/thm06_pareto_information/` | 5 amplitudes x 5 K; 1024 midpoint worlds, 101 targets, T=128; corrected discretization bounds saved |
| Fixed-regularity singleton | `Code/experiments/check_fixed_regularity_information.py` | `Code/results/fixed_regularity_information/` | Six deterministic widths, 1001-point decision/target grids; original records retained |
| Two-output identity | `Code/experiments/check_k2_coverage_identity.py` | `Code/results/fixed_k2_coverage/` | Five widths and exact endpoint/crossing calculation; original records retained |
| Local/global pair diagnostics | `Code/experiments/check_thm10_fixed_k2.py` | `Code/results/fixed_k2_minimax/` | 10201 local and 9452 nonlocal pairs; finite checks, not continuum proof |
| Nonquadratic one-query feedback | `Code/experiments/check_profile_class_feedback.py` | `Code/results/profile_class_feedback/` | beta=0/.5/1/2, four widths, 65 worlds, T=128; 315 quadratic identity checks; precision factors 0/.1/.5/2; prior grid separate |
| Full online synthetic suite | `Code/kaggle/run_synthetic.py`, `collect_results.py`, `plot_results.py`; `full.json` / redirected `release_full.json` | `reproducibility/result_jcam_synthetic_full.zip` | 2460 successful checkpoints, 246 ten-seed groups, seeds 0Ã¢â‚¬â€œ9; 2292 CUDA/168 CPU; environment and parity logs inside ZIP; E3 |
| Earlier EXP-10 variants | Experiment scripts under `Code/experiments/`, `Code/EXP10_REAL_DATA_REPORT.md` | `Code/results/exp10_bank/`, original public notebooks | Exploratory classification/controlled replay provenance, not final fixed-pool theorem evidence |

Original full synthetic output is intentionally retained as its existing archive, not duplicated as thousands of new files. README gives extraction, collection/plotting and optional new-run commands. C5 checks CRC, all seed groups and devices. Large Kaggle training is not repeated during release validation. Historical notebook builders referring to private proofs/manuscripts are not required by these commands.

## Bank source, licensing, metric and seeds

Official [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing), checked 2026-10-28, licenses sharing/adaptation under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: S. Moro, P. Rita and P. Cortez (2014), Bank Marketing, UCI, DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306). The existing source ZIP remains; preprocessing/changes are documented in `Code/data/bank_marketing/README.md` and `bank_stream.metadata.json`.

`prepare_bank_marketing.py` reads the date-ordered `bank-additional-full.csv` (41188 rows): prefix 8237, evaluation 32768=128x256, tail 183 discarded, age>=40 subgroup. Exact row timestamps are unavailable. The calibration subgroup has 4010 labels, 129 positives. The finite-pool experiment uses seeds 0Ã¢â‚¬â€œ4 and fractions .5/.8/.95, sampling `1000003*seed+1009*batch+int(fraction*100)` with batch IDs 1Ã¢â‚¬â€œ128. It evaluates continuous worst-target two-point coverage above the world-specific oracle floor. Labels yield `[k/N,(k+N-n)/N]`; log-loss is affine in the positive rate and the second objective is `(q-.01)^2` for q in [.01,.8]. The single interval is queried at the static right output; the adapted pair is then evaluated retrospectively on that same fixed pool. This is an after-query fixed-pool calculation, not cumulative online performance of the stationary theorem. Containment is deterministic for the current fixed pool, not predictive coverage for future observations. All 1920 records remain in the public original CSV. Dispersion is descriptive, with paired pool aggregation, not inference from 1920 independent samples.

## Integrity and clean validation

`reproducibility/final_jgo_sha256.json` lists SHA-256 hashes of essential scripts/configurations/raw data and all final outputs. Text hashes normalize CRLF to LF so Windows and Linux checkouts agree; binary hashes use exact bytes. The listed text extensions are recorded in the hash manifest. Its own hash and this Markdown manifest are excluded to avoid circular references. `python Code/validation/verify_final_release.py` verifies those hashes and original synthetic completeness. C5 additionally runs imports, unit tests, smoke, all 21 coefficient calculations, all 18 independent optimizations, table comparisons and computational figure creation. The exact clean-clone test outcome is recorded in `reproducibility/FINAL_RELEASE_VALIDATION.md`.

Limitations: exact Figure 1 typography is private manuscript composition; all diagram content is public. Table 5 is historical provenance, not a measured benchmark. Numerical checks support computational claims, not theorem proofs. No large GPU rerun or unsupported superpopulation inference is claimed.
