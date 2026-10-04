import numpy as np
from interval_pareto.objectives import QuadraticIntervalRound
from experiments.run import run_case
from interval_pareto.synthetic import SyntheticConfig


def test_endpoint_gradients_and_zero_width_limit():
    item = QuadraticIntervalRound(np.array([[0.2], [0.8]]), np.array([0.1, 0.1]),
                                  np.zeros((2, 1)), np.array([[0.1], [0.1]]))
    x, w = np.array([0.5]), np.array([0.5, 0.5])
    mid = item.midpoint_gradient(x, w)
    np.testing.assert_allclose(item.upper_gradient(x, w), mid + 0.1)
    np.testing.assert_allclose(item.lower_gradient(x, w), mid - 0.1)
    zero = QuadraticIntervalRound(item.centers, np.zeros(2), item.biases)
    np.testing.assert_allclose(zero.upper_gradient(x, w), zero.midpoint_gradient(x, w))
    np.testing.assert_allclose(zero.lower_gradient(x, w), zero.midpoint_gradient(x, w))


def test_lower_endpoint_driver_is_not_theorem_certified():
    config = SyntheticConfig(horizon=3, dimension=1, objectives=2,
                             radius=0.1, width_slope_fraction=0.5)
    summary, trace = run_case(config, eta=0.1, algorithm="lower")
    assert summary["bound_respected"] is None
    assert len(trace) == 3
