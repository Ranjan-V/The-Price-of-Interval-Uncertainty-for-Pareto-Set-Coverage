# APP-02: exact finite-pool Pareto coverage on Bank Marketing

**Run date:** 4 October 2026. **Source:** UCI Bank Marketing `bank-additional-full.csv`, 41,188 rows ordered by date, [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306), CC BY 4.0. The original archive SHA-256 is `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`. We use the first 8,237 rows to fix a reference pair and the next 32,768 rows as 128 chronological pools of 256 rows each.

The hidden target in each pool is the exact positive-label fraction among customers aged 40 or older. A decision `q∈[0.01,0.8]` is a constant positive-class probability. The two objectives are that group's Bernoulli log loss and the known resource penalty `(q-0.01)^2`. The method outputs two probabilities. It initially uses the calibration-prefix pair, receives one log-loss interval at its right output, inverts it, then outputs a revised pair for the **same** pool.

For a pool with `N` older-group labels, sampling `n` of them without replacement and seeing `k` positives gives the deterministic interval `p∈[k/N,(k+N-n)/N]`. Since log loss is affine in `p`, this interval contains the exact finite-pool objective at every decision, with no IID assumption or confidence parameter. The observation fractions are 50%, 80%, and 95%; five prespecified sampling seeds were used for each of 128 pools. One interval is reported after the initial pair, but producing it uses many label observations. The cost is stated explicitly.

The evaluation computes the **continuous** worst-target two-output coverage `sup_{u∈E} min_{x∈S} max_j[h_j(x)-h_j(u)]_+` by endpoint and scalar-crossing roots, then subtracts the exact world-specific two-point oracle floor. A separate dense-target diagnostic checked 25 random pairs; the continuous value exceeded the gridded value by less than `1e-4` and was never below it. The script passed all 1,920 finite-pool containment checks.

| Pool labels observed | Mean static excess | Mean adaptive excess | Adaptive / static | Fraction individually improved |
|---:|---:|---:|---:|---:|
| 50% | 0.175881 | 0.038674 | 0.2199 | 42.97% |
| 80% | 0.175881 | 0.009725 | 0.0553 | 60.78% |
| 95% | 0.175881 | 0.001385 | 0.0079 | 81.88% |

The mean static excess is large partly because late pools differ markedly from the early calibration prefix. The 50% interval improves fewer than half of individual cases despite a lower mean excess. At 95%, the interval is tighter, but nearly all older-group labels are observed. These are within-pool retrospective comparisons, not deployment or future-population claims. The hidden rate changes by batch and the objectives are asymmetric, so these data do not test the stationary minimax constants of THM-12.

Reproduce locally with `python Code/experiments/real_bank_pareto_coverage.py`. The script verifies the original archive against the prepared metadata when available; the compact Kaggle notebook carries the processed NPZ and UCI archive hash so it can run without internet. Numeric outputs and the figure are in `Code/results/real_bank_pareto_coverage/`. Proof of the interval and metric reductions is in `Math/proofs/proof_bank_finite_pool_coverage.md`.

[Kaggle committed version 1](https://www.kaggle.com/code/ranjanv1/jcam-thm12-bank-coverage/output?scriptVersionId=354990388) completed with `run_ok: True` and exposes `result_jcam_thm12_bank_coverage.zip` (543,572 bytes) on its Output tab. The compact notebook embeds the processed NPZ and source metadata, rather than the original raw UCI ZIP, to stay below Kaggle's 1 MB notebook-source limit.

## Independent numerical audit (4 October 2026)

`python Code/validation/audit_bank_oracle.py` independently recomputed all 1,920 fixed-pool rates and interval endpoints from the prepared labels. It evaluated the saved static and adaptive pairs on a separate 1,001-target grid: the largest continuous-minus-grid coverage difference was `3.25e-15`. A separate two-variable differential-evolution search at five representative observed pool rates returned oracle-floor estimates within `7.62e-5` of the reported scalar-root floors. That search is a numerical diagnostic, not a certified global-optimum proof. The saved audit is `Code/results/real_bank_pareto_coverage/independent_audit.json`. The rerun preserved all table values and added the output-pair coordinates to the per-run CSV for reproducibility. The local 16-test suite passes. Kaggle committed version 1 predates these additional audit columns; its reported results match the local rerun.
