import numpy as np
from interval_pareto.intervals import IntervalArray, lu_dominates, weak_lu_dominates
from interval_pareto.pareto import efficient_mask, lu_efficient_mask, additive_coverage, hausdorff


def test_interval_orders_and_width():
    a = IntervalArray(np.array([0.0, 1.0]), np.array([2.0, 2.0]))
    b = IntervalArray(np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    assert lu_dominates(a, b)
    assert not weak_lu_dominates(a, b)
    np.testing.assert_allclose(a.width, [2.0, 1.0])
    np.testing.assert_array_equal(lu_efficient_mask([a, b]), [True, False])


def test_finite_pareto_and_coverage():
    values = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(efficient_mask(values), [True, True, False])
    assert additive_coverage(values[:2], values[:2]) == 0.0
    assert additive_coverage(values[:2], values[:1]) == 1.0
    assert hausdorff(np.array([[0.0], [1.0]]), np.array([[0.0]])) == 1.0
