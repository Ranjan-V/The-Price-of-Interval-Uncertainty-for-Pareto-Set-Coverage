# Laptop CPU validation report — 2026-10-01

## Outcome

`LOCAL_TESTS_PASS` — 14 tests passed, 0 failed in the final suite. All runs used the Windows laptop CPU environment, including the smoke experiment; the local NVIDIA RTX 3050 was not used.

## Test progression

| Group | Result |
|---|---|
| Interval ordering and finite Pareto utilities | 2 passed |
| Variation, regret, synthetic containment and zero-width limit | 4 passed |
| Streaming surrogate, hidden-world identity and parallel front | 4 passed |
| Endpoint baseline tests added during validation | 2 passed |
| Kaggle checkpoint/resume and collection tests | 2 passed |
| Final full suite | 14 passed, 0 failed |

## Smoke and repeatability

The tiny scalar run used `T=64,d=2,m=2,seed=17`; its companion finite-front run used `T=24,d=1,m=2,K=5`. Both completed and serialized `summary.json`, `rounds.csv`, and a PNG diagnostic. Identical reruns in `results/smoke/run1/` and `run2/` have byte-identical summary and round-table SHA-256 hashes. Measured script-body runtimes ranged from 0.24 to 0.42 seconds across successful reruns; these are not benchmark results. Scalar regret was 0.29645 against the exact latent weighted comparator; THM-02 RHS was 47.69582. The finite-grid coverage diagnostic was 0.35815 against a THM-05 analytic bound of 228.37932. Both bounds are very loose on this tiny instance. The small `experiments/run.py` diagnostic driver also completed from `configs/smoke.json` and wrote result/round/metadata files.

## Sanity cases

All six SANITY-01–06 passed; machine-readable values are in `results/sanity/sanity.json`. Zero width yielded `Q_T=U_T=0` and exact midpoint/latent equality. Zero drift with fixed weights yielded `P_T=0`. Widths `[0,.01,.02,.05,.10]` over eight rounds produced `U_T=8×width` and the same `Q_T` for constant-width intervals. A comparator moved exactly `0.1` each of five transitions and the computed path was `0.5`. Hand-checkable scalar and LU dominance cases matched minimization order.

## Numerical theorem checks

`results/theorem_checks/` contains 17 successful small checks: 12 THM-02 instances across two seeds, three widths and two drift settings; two sampled-metric THM-05 checks; and three THM-03 two-world identities. The checks evaluate the exact THM-02 components separately, including `Q_T` and `P_T`. The THM-05 numerical metric samples the continuous front, so this is a diagnostic rather than a numerical proof of the continuous supremum. No bound violations occurred in these tested cases.

## Failure and fix

The first smoke attempt failed during `summary.json` serialization because NumPy boolean theorem flags were not native JSON booleans. Classified as a **code serialization bug**. Converted flags explicitly with `bool()` in scalar and front summaries, then reran both smoke cases successfully and verified identical hashes. No mathematical-definition mismatch was uncovered by local checks.

## Remaining limits

The full Kaggle pipeline was dry-run locally with a deliberately tiny configuration: stages A–H produced 192 per-seed synthetic checkpoints across 96 groups, one toy application checkpoint, tables and nine figures. This validates CPU packaging and resume behavior, not the large-scale experimental conclusions. The streaming-classification driver does not establish population interval containment and is not a theorem validation. The THM-05 experiment relies on the one-dimensional quadratic front characterized in LEM-01 and `K` gradient queries. Larger sweeps, Kaggle CUDA behavior, calibration on real data, and external novelty assessment remain untested.
