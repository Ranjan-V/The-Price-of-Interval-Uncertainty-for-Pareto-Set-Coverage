"""Deterministic toy input for serialization testing, not a real-data result."""

from pathlib import Path
import numpy as np


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "results" / "smoke" / "toy_stream.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(29)
    horizon, batch, dimension = 4, 6, 2
    features = rng.normal(size=(horizon, batch, dimension))
    labels = rng.choice(np.array([-1.0, 1.0]), size=(horizon, batch))
    groups = np.tile(np.array([0, 1, 0, 1, 0, 1]), (horizon, 1))
    radii = np.full((horizon, 3), 0.05)
    weights = np.tile(np.array([0.6, 0.3, 0.1]), (horizon, 1))
    np.savez(output, features=features, labels=labels, groups=groups,
             radii=radii, weights=weights)
    print(output)
