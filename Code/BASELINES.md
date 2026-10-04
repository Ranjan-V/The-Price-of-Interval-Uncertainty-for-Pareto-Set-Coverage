# Baselines and information budget

All online methods see the preference weight before play and receive only the corresponding endpoint-derived gradient **after** play. None receives latent values, the latent comparator, or future objectives. Latent values and exact comparators are offline evaluation oracles in the synthetic simulation.

| Internal label | Update gradient | Information / interpretation | Theorem |
|---|---|---|---|
| `midpoint` | weighted interval midpoint | Interval-center learner; it is the THM-02 algorithm. Width enters the analysis, not its update. | THM-02 |
| `upper` | weighted upper endpoint | Pessimistic interval-aware endpoint method. No latent-regret theorem asserted here. | None |
| `lower` | weighted lower endpoint | Optimistic endpoint method. No latent-regret theorem asserted here. | None |
| `static` | no update after initial `x=0.5` | Stationary-action ablation; not an online learning method. | None |
| `parallel midpoint` | one midpoint gradient for each fixed grid preference | `K` actions and `K` post-decision queries; set-output method. | THM-05 under A7–A10 |
| zero-width midpoint | same as `midpoint` with `L=U=h` | Ordinary exact-objective OGD limiting case. | THM-02 with `Q_T=0` |

The baseline names are internal descriptions, not claims of correspondence with named literature algorithms. Endpoint methods use the same synthetic environment, seed, horizon, and preference sequence for fair comparison. A comparison to a latent oracle is evaluation only and must never feed its information back to a learner. In the current synthetic family, widths are decision-dependent when `width_slope_fraction>0`; otherwise upper, midpoint, and lower gradients coincide, a real degeneracy that must be disclosed in plots.
