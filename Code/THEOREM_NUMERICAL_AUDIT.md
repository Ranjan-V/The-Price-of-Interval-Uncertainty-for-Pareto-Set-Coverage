# Numerical audit of proved claims — laptop CPU

**THM-02:** The code evaluates `R_T=Σ_t w_t·(h_t(x_t)-h_t(u_t))` and the exact RHS `(D²+2DP_T)/(2η)+ηG²T/2+Q_T`, with `D=√d`, `G=2√d`, unit-box projection, and `Q_T=(1/2)Σ_t w_t·[q_t(x_t)+q_t(u_t)]`. Twelve small midpoint-OGD cases passed `R_T≤RHS+10⁻⁹`; all also passed `Q_T≤U_T+10⁻⁹`. The two-location and uniform widths are separately stored. This does not validate endpoint baselines, which use different update gradients.

**THM-05:** Two one-dimensional parallel-grid cases passed sampled coverage ≤ the analytic continuous-front bound. The finite sample cannot rule out a missed peak between sampled efficient decisions; the mathematical theorem is established by proof, while this check catches implementation discrepancies. The experiment uses `K` actions and `K` post-decision midpoint gradients, as required.

**THM-03:** Three width settings passed the exact two-world identity `R_+ + R_-=U_T`, so the worse world's regret is at least `U_T/2`. This illustrates the algebra and does not empirically prove the minimax theorem.

No observed numerical violation. A violation in later experiments must halt scaling and trigger a review of feedback, comparator, interval containment, constants, and numerical precision before results are used. Detailed records: `results/theorem_checks/checks.csv` and `summary.json`.
