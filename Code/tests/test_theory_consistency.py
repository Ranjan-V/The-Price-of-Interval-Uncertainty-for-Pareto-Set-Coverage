import numpy as np
from interval_pareto.algorithms import MidpointOGD
from interval_pareto.objectives import QuadraticIntervalRound
from interval_pareto.regret import evaluate, scalar_static_regret, scalar_dynamic_regret
from interval_pareto.synthetic import SyntheticConfig, generate, weight_schedule
from interval_pareto.variation import path_variation, sampled_objective_variation


def test_zero_width_and_stationary_path():
    config = SyntheticConfig(horizon=8, dimension=1, objectives=2, drift=0.0, radius=0.0)
    rounds = generate(config)
    weights = np.tile([0.5, 0.5], (config.horizon, 1))
    learner = MidpointOGD(1, 0.1)
    actions = []
    for item, w in zip(rounds, weights):
        x = learner.act()
        actions.append(x)
        learner.observe_gradient(item.midpoint_gradient(x, w))
    comparator = np.asarray([item.latent_weighted_minimizer(w) for item, w in zip(rounds, weights)])
    report = evaluate(rounds, np.asarray(actions), comparator, weights, 0.1)
    assert report.two_location_width == 0.0
    assert report.uniform_width == 0.0
    assert report.path_variation == 0.0
    assert report.regret <= report.theorem_bound + 1e-10


def test_containment_and_nonconstant_width():
    config = SyntheticConfig(horizon=5, dimension=2, objectives=3, radius=0.2,
                             width_slope_fraction=0.5)
    for item in generate(config):
        for x in [np.zeros(2), np.ones(2), np.full(2, 0.5)]:
            interval = item.interval(x)
            latent = item.latent(x)
            assert np.all(interval.lower <= latent + 1e-12)
            assert np.all(latent <= interval.upper + 1e-12)


def test_path_variation_constant():
    assert path_variation(np.array([[0.2], [0.2], [0.2]])) == 0.0


def test_regret_and_sampled_variation_arithmetic():
    losses = np.array([2.0, 1.0])
    comparators = np.array([1.0, 0.5])
    assert scalar_static_regret(losses, comparators) == scalar_dynamic_regret(losses, comparators)
    assert scalar_dynamic_regret(losses, comparators) == 1.5
    table = np.array([[[0.0, 1.0]], [[2.0, 1.5]]])
    assert sampled_objective_variation(table) == 2.0
