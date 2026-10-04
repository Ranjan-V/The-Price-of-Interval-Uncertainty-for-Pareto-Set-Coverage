import numpy as np
from interval_pareto.algorithms import ParallelMidpointOGD, simplex_lattice
from interval_pareto.synthetic import SyntheticConfig, generate
from experiments.parallel_front import run_front_case


def test_simplex_grid_and_parallel_feedback_shapes():
    weights = simplex_lattice(2, 4)
    assert len(weights) == 5
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    item = generate(SyntheticConfig(horizon=1, dimension=1, objectives=2))[0]
    learner = ParallelMidpointOGD(weights, 1, 0.1)
    actions = learner.act()
    assert actions.shape == (5, 1)
    learner.observe(item, actions)


def test_small_front_coverage_respects_analytic_bound():
    config = SyntheticConfig(horizon=4, dimension=1, objectives=2, drift=0.1,
                             radius=0.1, width_slope_fraction=0.25)
    report = run_front_case(config, denominator=2, eta=0.1, front_grid=11)
    assert report["finite_grid_below_bound"]
