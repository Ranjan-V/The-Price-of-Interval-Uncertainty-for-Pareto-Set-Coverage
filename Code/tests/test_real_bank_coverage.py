"""Independent gridded checks for the continuous real-data coverage evaluator."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from real_bank_pareto_coverage import Q_MIN, coverage, feedback_interval, oracle_pair, resource, risk


def test_continuous_coverage_against_dense_target_grid():
    rng = np.random.default_rng(99)
    for _ in range(25):
        p = float(rng.uniform(0.01, 0.7))
        pair = tuple(sorted(rng.uniform(0.01, 0.8, 2)))
        exact = coverage(pair, p)
        sampled = max(
            min(max(resource(x) - resource(float(u)),
                    risk(x, p) - risk(float(u), p), 0.0) for x in pair)
            for u in np.linspace(Q_MIN, p, 2001)
        )
        assert sampled <= exact + 1e-11
        assert exact - sampled < 1e-4


def test_oracle_and_exact_finite_pool_containment():
    for p in (0.0, 0.02, 0.05, 0.1, 0.3, 0.65):
        pair, floor = oracle_pair(p)
        assert abs(coverage(pair, p) - floor) < 1e-9
        assert floor <= coverage((Q_MIN, max(Q_MIN, p)), p) + 1e-10
    # Four observed positives among 17 sampled labels from a 20-label pool.
    # All three possible remaining positive counts must be covered.
    lo, hi, _, _ = feedback_interval(0.07, 4, 17, 20)
    for missing in range(4):
        assert lo - 1e-12 <= risk(0.07, (4 + missing) / 20) <= hi + 1e-12
