# Continuous-metric correction for the THM-06/07 numerical check

The CSV is produced on `N=1024` midpoint worlds in `I=[r,1-r]`, `r=1/20`, and `M=101` equally spaced target points in each true efficient interval. A sampled supremum can miss a larger continuous value. The summary therefore reports corrected bounds as well as raw measurements.

For fixed hidden world and output set, the coverage integrand as a function of efficient target `u` is `2a`-Lipschitz. The nearest sampled target is at distance at most `r/(M-1)`, so continuous coverage is between sampled coverage and sampled coverage plus `δ_u=2ar/(M-1)`.

For the fixed blind grid, sampled coverage as a function of `θ` is also `2a`-Lipschitz. Midpoint quadrature over an interval of length `L=1-2r` therefore has mean error at most `δ_mean=aL/(2N)`. The oracle grid translates with `θ`, so its coverage is exactly translation invariant. Thus the continuous uniform-world expected excess of the blind grid over the *optimal* world-specific `K`-set is at least

`mean(sampled blind coverage - sampled oracle-grid coverage) - δ_u - δ_mean`.

The universal grid's continuous maximum over both `θ` and `u` is at most

`max(sampled universal-grid coverage) + δ_u + δ_max`, where `δ_max=aL/N` accounts for a hidden world between sampled midpoints.

The corrected lower estimate exceeded `a/(80K)` and the corrected universal-grid upper estimate stayed below `a/K` for all 25 combinations of `a` and `K`. The smallest corrected lower/theorem ratio was 1.924. These are verified discretization statements about this construction; the analytic proofs of THM-06/07 do not rely on numerical evaluation.
