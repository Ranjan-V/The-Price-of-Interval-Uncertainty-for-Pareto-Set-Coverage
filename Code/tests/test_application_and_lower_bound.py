import numpy as np
from interval_pareto.applications.streaming import ClassificationBatch, objective_and_subgradients


def test_streaming_shapes_and_convex_surrogates():
    batch = ClassificationBatch(np.array([[1.0, 0.0], [0.0, 1.0]]),
                                np.array([1.0, -1.0]), np.array([0, 1]))
    values, gradients = objective_and_subgradients(batch, np.array([0.2, 0.4]))
    assert values.shape == (3,)
    assert gradients.shape == (3, 2)
    assert np.all(np.isfinite(values))


def test_two_world_identity():
    widths = np.array([0.1, 0.2, 0.3])
    actions = np.array([0.0, 0.5, 1.0])
    plus = float(widths @ actions)
    minus = float(widths @ (1 - actions))
    assert np.isclose(plus + minus, widths.sum())
