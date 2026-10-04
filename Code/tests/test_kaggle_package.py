import json
from interval_pareto.synthetic import SyntheticConfig
from kaggle.common import canonical_id, valid_checkpoint
from kaggle.run_synthetic import run_matrix
from kaggle.collect_results import collect


def test_checkpoint_identity_is_key_order_invariant(tmp_path):
    assert canonical_id({"a": 1, "b": 2}) == canonical_id({"b": 2, "a": 1})
    assert not valid_checkpoint(tmp_path / "missing.json", "anything")


def test_tiny_synthetic_resume_and_collection(tmp_path):
    base = SyntheticConfig(horizon=3, dimension=1, objectives=2, drift=0,
                           radius=0.01, seed=1)
    config = {"base": {k: v for k, v in vars(base).items() if k != "seed"},
              "eta": 0.1, "seeds": [1], "widths": [0.0], "drifts": [0.0],
              "horizons": [3], "joint_widths": [0.0], "joint_drifts": [0.0],
              "objective_counts": [2], "dimensions": [1], "front_denominator": 2}
    first = run_matrix(config, ["EXP-01"], "one", "cpu", tmp_path, max_jobs=1)
    second = run_matrix(config, ["EXP-01"], "one", "cpu", tmp_path, max_jobs=1)
    assert first["completed"] == 1 and first["failed"] == 0
    assert second["completed"] == 0 and second["skipped"] == 1
    rows, aggregate = collect(tmp_path)
    assert len(rows) == len(aggregate) == 1
    assert rows[0]["regret"] is not None
