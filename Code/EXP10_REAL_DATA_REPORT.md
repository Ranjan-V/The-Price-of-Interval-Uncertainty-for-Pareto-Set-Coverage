# EXP-10: chronological Bank Marketing application

## Execution

Kaggle [JCAM_EXP10_Bank_Marketing, version 2](https://www.kaggle.com/code/ranjanv1/jcam-exp10-bank-marketing?scriptVersionId=354425845) completed all five imported notebook cells in about 22 seconds on CPU. The private [JCAM EXP10 Bank Marketing Stream](https://www.kaggle.com/datasets/ranjanv1/jcam-exp10-bank-marketing-stream) input includes the UCI download, the prepared NPZ, preprocessing and evaluation code, metadata, and attribution. `D:/JCAM/result_jcam_exp10.zip` passed ZIP integrity validation and its `notebook_stages.json` records `run_ok: true`.

## Data and protocol

Source: UCI Bank Marketing `bank-additional-full.csv`, 41,188 examples in published date order (May 2008–November 2010), CC BY 4.0, by Moro, Rita, and Cortez. The source archive SHA-256 is `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`. The UCI file does not provide exact per-row timestamps. The first 8,237 rows initialize causal feature scaling and the prevalence baseline. The next 32,768 rows form 128 consecutive batches of 256; 183 tail rows are unused. The label is subscription yes/no. The group is age below 40 versus at least 40; age is not a model feature. Every batch contains both groups (minimum counts 45 and 44). Post-call duration is excluded to prevent prediction leakage.

Seven pre-call numeric variables are mapped using only prior rows, clipped to a fixed signed range and represented with positive/negative feature pairs. Five positive/negative bias pairs allow a negative intercept within the learner's `[0,1]^d` decision box. The resulting decision dimension is 24. The three objective weights are `(0.80, 0.15, 0.05)` for logistic loss, absolute group mean score gap, and quadratic resource cost. Supplied radii `(0.05, 0.05, 0)` are fixed sensitivity bands. They have **no verified population coverage** and do not affect midpoint actions in the current application driver. The learning rate is the predeclared `0.05` from the full experiment configuration; it was not tuned on evaluation data.

## Prequential results

Actions were selected before each batch's labels were consumed. A causal comparator predicts the positive class frequency observed in the initial prefix and earlier batches only.

| Metric | Online midpoint | Causal prevalence baseline |
|---|---:|---:|
| Overall log loss | 0.35897 | 0.40205 |
| Accuracy at 0.5 threshold | 0.86200 | 0.86914 |
| First 64 batches log loss | 0.25780 | 0.21994 |
| Last 64 batches log loss | 0.46015 | 0.58417 |

The observed positive rate rose from 0.0565 in the first half to 0.2052 in the second half. The online model had lower batch log loss in 47 of 128 batches, but the later gains made its overall log loss lower. Its accuracy was lower. Mean absolute age-group score gap was 0.00626 across batches, with a maximum of 0.14684; a small mean must not be read as a uniform fairness guarantee. Recomputed pre-update logistic losses matched the stored round objectives exactly.

## Interpretation and limits

This demonstrates a real chronological empirical surrogate application, not theorem validation. The radii lack calibrated population containment, the age-group score gap is a surrogate rather than a guarantee on outcomes, and date order without exact timestamps limits temporal analysis. There is only one predeclared source and one learning rate. The next scientific steps are radius calibration under an explicit population/feedback model, additional online baselines and preference settings, and uncertainty analysis for paired performance differences. No real-data theorem claim is made.

## Chronological audit follow-up

Kaggle [JCAM_EXP10_Bank_Marketing, version 3](https://www.kaggle.com/code/ranjanv1/jcam-exp10-bank-marketing?scriptVersionId=354429520) completed all five imported code cells in 23 seconds. Its saved output page has `result_jcam_exp10_audit.zip`; the notebook printed `run_ok: True`. `Code/experiments/audit_bank_stream.py` replays the uploaded action sequence exactly (`max_replay_action_error = 0`) and compares four box-projected online learners on the same 128 batches and learning rate. Each uses only its own preceding gradients. The alternatives vary the weights on logistic loss, absolute age-group mean score gap, and quadratic resource cost. These weights express different preferences, so this is a tradeoff analysis rather than a claim of one method dominating another.

| Preference weights | Log loss | Mean score gap | Mean resource cost |
|---|---:|---:|---:|
| Original (0.80, 0.15, 0.05) | 0.35897 | 0.00626 | 2.42034 |
| Logistic only (1, 0, 0) | 0.35326 | 0.00734 | 3.25907 |
| Gap heavy (0.65, 0.30, 0.05) | 0.36578 | 0.00581 | 2.40121 |
| Resource heavy (0.70, 0.10, 0.20) | 0.36502 | 0.00495 | 1.21441 |

The audit defines a **new, observable target**: the next batch's empirical objective at the action chosen before that batch. It forecasts this target by evaluating the same action on the previous observed batch. After 20 warmup batches, a radius for each objective is computed from the expanding history of absolute one-step forecast errors using the nominal 90% order statistic. No future batch is used to forecast or size its own interval. This rule is a diagnostic under temporal drift, not a valid conformal or population guarantee.

Across 108 evaluated batches, empirical marginal containment was 84.3% for logistic loss and 83.3% for score gap; joint containment was 71.3%. Both nontrivial marginal rates fall below the nominal 90% level. Mean calibrated radii were 0.11974 and 0.00684, respectively. The old fixed 0.05 bands, if incorrectly treated as forecast radii, would contain 64.8% of logistic losses and 95.4% of score gaps. Resource cost is deterministic at a fixed action and has zero one-step forecast error, so its zero radius contains all observations.

This audit does **not** establish containment of a latent population objective, sequential validity, fairness of outcomes, or a theorem assumption. Any such claim needs an explicit stochastic target and assumptions about sampling or drift. The next step is to develop and stress-test that target and an interval construction suited to it, then add uncertainty analysis for paired performance differences.

## Conditional population-band proposition

Kaggle [JCAM_EXP10_Bank_Marketing, version 4](https://www.kaggle.com/code/ranjanv1/jcam-exp10-bank-marketing?scriptVersionId=354447179) completed all five imported cells in 18 seconds and saved `result_jcam_exp10_population_bands.zip` with `run_ok: True`. The ZIP includes the proof and per-batch width table. The separate [APP-01 proof](../Math/streaming_population_intervals.md) defines conditional population logistic loss and age-group score gap for each batch. It assumes that, given all prior batches, the 256 examples *within the current batch* are IID from one distribution. The distribution may drift arbitrarily between batches. Hoeffding bounds, a finite grid over the effective eight score parameters, and group-coordinate concentration yield 95% simultaneous containment over all 128 rounds, all objectives, and all box actions **if this sampling assumption holds**. The UCI date ordering does not verify it. The actual population objectives are unobserved, so empirical population coverage cannot be measured from this dataset.

`Code/experiments/certify_bank_population.py` computes the resulting widths from the EXP-10 batch and group sizes. The uniform logistic half-width is **3.7691** versus observed mean loss **0.3590**. The mean age-score-gap half-width is **6.2222** versus observed mean gap **0.00626**. Resource cost remains exact. The THM-02 uncertainty-width upper bound `U_T` is **1010.84**; adding its static-comparator OGD part at `η=0.05` gives a **1340.28** cumulative envelope over 128 batches. This is a conditional worst-case bound, not measured regret, and it is too loose to support a useful real-data performance claim. Sharper distributional assumptions, independent repeated batches, or a different target would be required for informative population bands.

## Controlled sampling check

Kaggle [JCAM_EXP10_Bank_Marketing, version 5](https://www.kaggle.com/code/ranjanv1/jcam-exp10-bank-marketing?scriptVersionId=354448435) completed all six imported cells in 26 seconds, saved `result_jcam_exp10_controlled_resampling.zip`, and printed `run_ok: True`. `Code/experiments/controlled_bank_resampling.py` treats each original chronological 256-row window as a fixed, known finite population and draws a fresh 256-row feedback batch **with replacement** from that window. This makes the within-round IID assumption true by design, while the window populations can drift chronologically. The learner sees only sampled feedback; the original window is used solely as an offline population oracle. Ten predeclared random seeds produce 1,280 online rounds. At every played action, the conditional population bands contained both risk and age-gap objectives in all ten runs. The largest observed risk and gap errors were 0.1753 and 0.2001, respectively, far smaller than the uniform radii of 3.7691 and about 6.23. Fixed 0.05 bands contained only 88.0% of risk values and 99.5% of gap values on average. These checks evaluate played actions; the proof, rather than finite testing, supplies uniformity over all actions. This designed experiment is distinct from the original observational UCI stream and does not validate its IID assumption.

## Sharper uniform risk certificate

A one-sided Rademacher and bounded-difference argument now supplements the original grid-Hoeffding construction in [APP-01](../Math/streaming_population_intervals.md). The certified logistic half-width falls from **3.7691** to **3.1115** while retaining simultaneous 95% coverage over all 128 rounds and all box actions under the same conditional within-batch IID assumption. The group-gap construction is unchanged. Re-running the controlled ten-seed check leaves all 1,280 played actions covered. The THM-02 width term falls from 1010.84 to **876.18**; its static-comparator envelope at the predeclared learning rate falls from 1340.28 to **1205.61**. This remains too large to provide an informative performance guarantee. The original observational stream still lacks a verified IID sampling design. Local numeric results are in `Code/results/exp10_bank/population_certificate_sharper_local/` and `Code/results/exp10_bank/controlled_resampling_sharper_local/`.

Kaggle [version 6](https://www.kaggle.com/code/ranjanv1/jcam-exp10-bank-marketing?scriptVersionId=354485355) completed the six imported code cells in 22.8 seconds with `run_ok: True` and `result_jcam_exp10_sharper_bands.zip` (121,191 bytes). The notebook printed the same 3.1115 radius, 876.18 width term, 1205.61 envelope, and ten-seed coverage as the local run.

## Predictable neutral-comparator certificate

Kaggle [version 7](https://www.kaggle.com/code/ranjanv1/jcam-exp10-bank-marketing/output?scriptVersionId=354487204) completed all seven imported code cells in 27 seconds and saved `result_jcam_exp10_predictable_comparator.zip` (132,310 bytes) with `run_ok: True`. The downloaded archive passed ZIP integrity validation and has six passed stages. The [conditional proof](../Math/predictable_comparator_intervals.md) bounds population regret against the **predeclared neutral action** `(0.5,...,0.5)` using intervals only at the learner's past-measurable played actions. Its risk and group-gap radii adapt to the played action's effective score coefficients; the neutral action's objectives are exact. This supports a narrower comparator claim than the all-action, hindsight-comparator THM-02 result.

For the original chronological stream, the computed uncertainty transfer is **58.09**, the realized-gradient optimization term **0.62**, and their sum **58.71** over 128 rounds. The observed empirical cumulative weighted action-minus-neutral difference is **-37.81**. The numerical 58.71 bound is **conditional on within-batch IID sampling**, which the observational UCI chronology does not establish; it is not an empirically verified population-regret bound on that stream. In a designed IID-with-replacement replay from each fixed 256-row chronological pool, ten predeclared seeds covered both objectives at all 128 played actions, and every realized finite-pool regret was below its calculated bound. These checks do not establish 95% coverage empirically or extend the certificate to a hindsight-chosen comparator.
