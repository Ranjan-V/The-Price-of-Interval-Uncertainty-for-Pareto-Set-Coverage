"""Optional Torch batch over independent seeds; time recurrence stays sequential.

Only the midpoint THM-02 algorithm is implemented. All tensors use float64.
Kaggle Stage B must compare this path with the NumPy CPU path before full runs.
"""

import numpy as np


def evaluate_batch(configs: list, weights: np.ndarray, eta: float, device: str) -> list[dict]:
    import torch
    from interval_pareto.synthetic import generate

    if not configs or any((c.horizon, c.dimension, c.objectives) !=
                          (configs[0].horizon, configs[0].dimension, configs[0].objectives) for c in configs):
        raise ValueError("matching nonempty synthetic configs required")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rounds_by_seed = [generate(config) for config in configs]
    centers = torch.as_tensor(np.asarray([[r.centers for r in rounds] for rounds in rounds_by_seed]),
                              dtype=torch.float64, device=device)
    radii = torch.as_tensor(np.asarray([[r.radii for r in rounds] for rounds in rounds_by_seed]),
                            dtype=torch.float64, device=device)
    biases = torch.as_tensor(np.asarray([[r.biases for r in rounds] for rounds in rounds_by_seed]),
                             dtype=torch.float64, device=device)
    slopes = torch.as_tensor(np.asarray([[r.width_slopes for r in rounds] for rounds in rounds_by_seed]),
                             dtype=torch.float64, device=device)
    w = torch.as_tensor(np.asarray(weights, float), dtype=torch.float64, device=device)
    if w.shape != (configs[0].objectives,) or torch.any(w < 0) or not torch.isclose(w.sum(), torch.tensor(1.0, dtype=torch.float64, device=device)):
        raise ValueError("fixed simplex weight required")
    b_count, horizon, m, d = centers.shape
    x = torch.full((b_count, d), 0.5, dtype=torch.float64, device=device)
    previous_u = None
    regret = torch.zeros(b_count, dtype=torch.float64, device=device)
    midpoint_regret = torch.zeros_like(regret)
    path = torch.zeros_like(regret)
    q_total = torch.zeros_like(regret)
    u_total = torch.zeros_like(regret)
    for t in range(horizon):
        c, r, b, s = centers[:, t], radii[:, t], biases[:, t], slopes[:, t]
        r_min = r - 0.5 * torch.sum(torch.abs(s), dim=2)
        weighted_center = torch.sum(w[None, :, None] * c, dim=1)
        latent_shift = torch.sum(w[None, :, None] * r_min[:, :, None] * b, dim=1) / d
        u = torch.clamp(weighted_center - latent_shift, 0.0, 1.0)
        if previous_u is not None:
            path += torch.linalg.vector_norm(u - previous_u, dim=1)
        previous_u = u
        midpoint_x = torch.sum((c - x[:, None, :]) ** 2, dim=2)
        midpoint_u = torch.sum((c - u[:, None, :]) ** 2, dim=2)
        tilt_x = torch.sum(b * (2.0 * x[:, None, :] - 1.0), dim=2) / d
        tilt_u = torch.sum(b * (2.0 * u[:, None, :] - 1.0), dim=2) / d
        regret += torch.sum(w[None, :] * (midpoint_x + r_min * tilt_x - midpoint_u - r_min * tilt_u), dim=1)
        midpoint_regret += torch.sum(w[None, :] * (midpoint_x - midpoint_u), dim=1)
        radius_x = r + torch.sum(s * (x[:, None, :] - 0.5), dim=2)
        radius_u = r + torch.sum(s * (u[:, None, :] - 0.5), dim=2)
        q_total += torch.sum(w[None, :] * (radius_x + radius_u), dim=1)
        u_total += torch.max(2.0 * (r + 0.5 * torch.sum(torch.abs(s), dim=2)), dim=1).values
        gradient = 2.0 * (x - weighted_center)
        x = torch.clamp(x - eta * gradient, 0.0, 1.0)
    diameter = d ** 0.5
    online = diameter ** 2 / (2.0 * eta)
    gradient_term = eta * (2.0 * diameter) ** 2 * horizon / 2.0
    records = []
    for i in range(b_count):
        p = float(path[i].item())
        q = float(q_total[i].item())
        drift = diameter * p / eta
        records.append({"regret": float(regret[i].item()),
                        "midpoint_regret": float(midpoint_regret[i].item()),
                        "path_variation": p, "two_location_width": q,
                        "uniform_width": float(u_total[i].item()),
                        "online_term": online, "drift_term": drift,
                        "gradient_term": gradient_term,
                        "theorem_bound": online + drift + gradient_term + q,
                        "finite_grid_coverage": None, "bound_respected": bool(float(regret[i].item()) <= online + drift + gradient_term + q + 1e-8)})
    return records
