"""Direct worst-target K=2 coverage for nonquadratic profiles and interval feedback.

This is a deterministic diagnostic, not a certificate for THM-11's compactness gap.
The value interval is deliberately one-sided, so its midpoint can err by ν.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Code" / "results" / "profile_class_feedback"
OUT.mkdir(parents=True, exist_ok=True)
R = 0.125
C = 0.5
T = 128


def phi(v: float, beta: float) -> float:
    return v * v + beta * v**4


def dphi(v: float, beta: float) -> float:
    return 2 * v + 4 * beta * v**3


def oracle_half_gap(beta: float) -> float:
    lo, hi = 0.0, R
    for _ in range(65):
        h = (lo + hi) / 2
        balance = phi(R - h, beta) - phi(0, beta) - phi(R + h, beta) + phi(R, beta)
        if balance > 0:
            lo = h
        else:
            hi = h
    return (lo + hi) / 2


def coverage(theta: float, left: float, right: float, beta: float) -> float:
    """Continuous worst-target metric: endpoints and unique crossing, bisection solved."""
    assert theta - R - 1e-12 <= left <= right <= theta + R + 1e-12
    a, b = left - theta, right - theta
    endpoint_left = phi(R + a, beta)
    endpoint_right = phi(R - b, beta)
    lo, hi = a, b
    for _ in range(65):
        v = (lo + hi) / 2
        loss_left = phi(R - a, beta) - phi(R - v, beta)
        loss_right = phi(R + b, beta) - phi(R + v, beta)
        if loss_left < loss_right:
            lo = v
        else:
            hi = v
    v = (lo + hi) / 2
    crossing = phi(R - a, beta) - phi(R - v, beta)
    return max(endpoint_left, endpoint_right, crossing)


rows = []
prior_grid_rows = []
quadratic_identity_checks = 0
maximum_identity_error = 0.0
for iz in range(-10, 11):
    z = iz * R / 80
    for ih in range(1, 16):
        h = ih * R / 50
        got = coverage(C, C + z - h, C + z + h, 0.0)
        expected = max((R - h + abs(z)) ** 2, (2 * R * h + h * h) * (1 - z * z / R**2))
        maximum_identity_error = max(maximum_identity_error, abs(got - expected))
        quadratic_identity_checks += 1
assert maximum_identity_error < 1e-12

for beta in (0.0, 0.5, 1.0, 2.0):
    hstar = oracle_half_gap(beta)
    e_star = phi(R - hstar, beta)
    if beta == 0:
        assert abs(hstar - R / 4) < 1e-12
        assert abs(e_star - 9 * R * R / 16) < 1e-12
    L = dphi(1.0, beta)
    for s in (R / 64, R / 32, R / 16, R / 8):
        width = phi(5 / 8 + s, beta) - phi(5 / 8 - s, beta)
        m_s = dphi(R + hstar - s, beta)
        smallest_prior_excess = float("inf")
        for im in range(11):
            center = C - s + 2 * s * im / 10
            for ih in range(11):
                half_gap = hstar - s + 2 * s * ih / 10
                prior_excess = (
                    coverage(C - s, center - half_gap, center + half_gap, beta)
                    + coverage(C + s, center - half_gap, center + half_gap, beta)
                ) / 2 - e_star
                smallest_prior_excess = min(smallest_prior_excess, prior_excess)
        assert smallest_prior_excess > 0
        prior_grid_rows.append({
            "beta": beta,
            "s": s,
            "pair_grid_checks": 121,
            "smallest_endpoint_prior_excess": smallest_prior_excess,
            "smallest_prior_excess_over_width": smallest_prior_excess / width,
        })
        world_results = {factor: [] for factor in (0.0, 0.1, 0.5, 1.0)}
        blind_results = []
        for i in range(65):
            theta = C - s + 2 * s * i / 64
            static_excess = coverage(theta, C - hstar, C + hstar, beta) - e_star
            assert -1e-12 <= static_excess <= L * s + 1e-12
            blind_results.append(T * max(0.0, static_excess))
            value = phi(R + hstar + C - theta, beta)
            for factor in world_results:
                nu = factor * m_s * s
                # A containing interval [value, value+2ν] has the largest possible midpoint bias.
                observed_midpoint = min(
                    max(value + nu, phi(R + hstar - s, beta)),
                    phi(R + hstar + s, beta),
                )
                lo, hi = C - s, C + s
                for _ in range(65):
                    mid = (lo + hi) / 2
                    if phi(R + hstar + C - mid, beta) > observed_midpoint:
                        lo = mid
                    else:
                        hi = mid
                estimate = (lo + hi) / 2
                assert abs(estimate - theta) <= nu / m_s + 1e-12
                adaptive_excess = coverage(theta, estimate - hstar, estimate + hstar, beta) - e_star
                total = max(0.0, static_excess) + (T - 1) * max(0.0, adaptive_excess)
                bound = L * s + (T - 1) * L * nu / m_s
                assert total <= bound + 1e-9
                world_results[factor].append(total)
        worst_blind = max(blind_results)
        for factor, totals in world_results.items():
            worst_feedback = max(totals)
            rows.append(
                {
                    "beta": beta,
                    "s": s,
                    "nu_over_m_s_s": factor,
                    "h_star": hstar,
                    "oracle_floor": e_star,
                    "common_width": width,
                    "U_T": T * width,
                    "worst_blind_excess": worst_blind,
                    "worst_feedback_excess": worst_feedback,
                    "feedback_to_blind": worst_feedback / worst_blind,
                    "feedback_bound": L * s + (T - 1) * L * factor * s,
                }
            )

with (OUT / "profile_class_feedback.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT / "profile_class_prior_grid.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(prior_grid_rows[0]))
    writer.writeheader()
    writer.writerows(prior_grid_rows)
summary = {
    "world_grid_size": 65,
    "horizon": T,
    "profiles": [0.0, 0.5, 1.0, 2.0],
    "widths_per_profile": 4,
    "feedback_levels_per_width": 4,
    "quadratic_identity_checks": quadratic_identity_checks,
    "maximum_quadratic_identity_error": maximum_identity_error,
    "rows": len(rows),
    "endpoint_prior_grid_checks": sum(row["pair_grid_checks"] for row in prior_grid_rows),
    "endpoint_prior_excess_over_width_range": [
        min(row["smallest_prior_excess_over_width"] for row in prior_grid_rows),
        max(row["smallest_prior_excess_over_width"] for row in prior_grid_rows),
    ],
    "zero_noise_feedback_to_blind_range": [
        min(row["feedback_to_blind"] for row in rows if row["nu_over_m_s_s"] == 0),
        max(row["feedback_to_blind"] for row in rows if row["nu_over_m_s_s"] == 0),
    ],
    "all_checks_passed": True,
    "qualification": "One-sided interval diagnostics; THM-11 global gap is proved by compactness, not this grid.",
}
(OUT / "profile_class_feedback_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
fig, ax = plt.subplots(figsize=(6.2, 3.7))
sample = [row for row in rows if row["beta"] == 1.0]
widths = sorted({row["s"] for row in sample})
blind = [next(row["worst_blind_excess"] for row in sample if row["s"] == s) for s in widths]
ax.plot(widths, blind, "o-", label="Common envelopes: static pair")
for factor in (0.0, 0.1, 0.5):
    values = [
        next(row["worst_feedback_excess"] for row in sample if row["s"] == s and row["nu_over_m_s_s"] == factor)
        for s in widths
    ]
    ax.plot(widths, values, "o-", label=rf"Informative interval: $\nu/(m_s s)={factor:g}$")
ax.set(xlabel="Hidden-world radius $s$", ylabel="Worst-world cumulative excess coverage")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "profile_class_feedback.png", dpi=200)
fig.savefig(OUT / "profile_class_feedback.pdf")
plt.close(fig)
print(json.dumps(summary, indent=2))
