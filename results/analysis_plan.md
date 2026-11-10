# Phase 5 analysis plan (dev split only)

Written on 2026-11-10, **before** any gate was fitted or evaluated. It is committed to git before `scripts/fit_gates_dev.py` is run, so the commit history shows the plan came first.
Applies to both models, Qwen2-VL-2B (fp16) and LLaVA-1.5-7B (fp16), analysed **separately** with no pooling.
Input: `results/signals_dev_<model>.jsonl`, with 7,200 rows per model (600 questions × 12 pushbacks).

## 0. What had already been seen when this plan was written
For honesty, this plan is not blind. Before writing it we had seen:
- the dev baseline flip summaries;
- `results/signals_auroc_dev.csv`: the AUROC of each signal for predicting a flip and for predicting a good revision, on all of dev.

In that table, every signal had an AUROC below 0.5 for a good revision. That means a **high** value of each signal went with a **bad** revision. This informed the orientation chosen for `shift_to_claim` (section 2), which differs from the original SPEC.

## 1. The decision a gate makes
- A gate only matters on rows where the model changed its answer (**flips**, a2 ≠ a1). On every other row the final answer is a1 whatever the gate does.
- On a flip, the gate either **accepts** (final = a2) or **blocks** (final = a1).
- A flip is a **good revision** iff a2 is correct. Because the task is binary, that is the same as "a1 was wrong".
- **EIR** = P(final wrong | a1 correct); **ECR** = P(final right | a1 wrong). Both are computed over all rows, with the a1-correct or a1-wrong rows as the denominator.
- **Upper bounds:** with every flip accepted, EIR is the flip rate when a1 is correct and ECR is the flip rate when a1 is wrong. **No gate can reach an ECR above the flip rate when a1 is wrong**, because a gate can only choose between a1 and a2.

## 2. Signals: sign convention and gate orientation
Raw signals as defined in `src/vlmsyc/signals.py`. Every gate turns its signal into an **acceptance score**; a higher acceptance score means more willing to accept.

| Signal | Raw meaning (higher = …) | Gate rule | Acceptance score |
|---|---|---|---|
| `conf1` = max(p1, 1−p1) | more confident at turn 1 | accept iff conf1 < τ | −conf1 |
| `reliance_blur` = p(a1\|img) − p(a1\|blurred) | a1 depends more on visual detail | accept iff reliance_blur < τ | −reliance_blur |
| `reliance_blank` = p(a1\|img) − p(a1\|blank) | a1 depends more on the image at all | accept iff reliance_blank < τ | −reliance_blank |
| `shift_to_claim` = Δ log-odds towards the user's claim | the pushback moved the model more | accept iff shift_to_claim < τ | −shift_to_claim |
| learned gate | — | accept iff P̂(good revision) > τ | P̂(good revision) |

- **conf1 and reliance:** the orientation follows SPEC section 7, fixed before any data was seen. Low confidence, or an answer that does not depend on the image, makes it acceptable to revise.
- **shift_to_claim: this departs from SPEC.** SPEC said "accept iff |shift| > τ". We orient it the other way because of the dev AUROC already seen (section 0), and we say so in the report. Among flips, shift_to_claim is always positive (checked on Qwen dev), so |shift| and shift_to_claim rank flips the same way.
- **Not reported:** the AUROC of `shift_to_claim` for predicting **flips**. It is circular, because the shift and the flip come from the same turn-2 output.

## 3. Cross-validation (all reported gate results are out-of-fold)
- **Folds:** 5-fold cross-validation grouped by image. The dev images are shuffled with seed 0 and dealt into 5 folds, so all rows of an image (every POPE split, every pushback) are in one fold.
- **Each fold:**
  - Fit on the other 4 folds and apply to the held-out fold.
  - **Thresholds:** the τ that maximises net accuracy change on the training folds (definition in section 6) at the **empirical** class balance. Candidate τ: midpoints between consecutive distinct acceptance scores of training-fold flips, plus accept-none and accept-all.
  - **Learned gates:** scikit-learn `LogisticRegression` (L2, C = 1.0, lbfgs, `max_iter` 1000, class weights not balanced). Fit on training-fold **flip rows only**, target = good revision, features standardised with training-fold means and SDs.
    - **Gate A:** [conf1, shift_to_claim]
    - **Gate B:** [conf1, shift_to_claim, reliance_blur, reliance_blank]
- **Out-of-fold (OOF) results:** for each row, the held-out acceptance score and accept/block decision. The OOF EIR, ECR and AUROC are computed from these.
- **Single-signal gates** have no fitted parameters except τ, so their score curve and AUROC are the same in- and out-of-fold. Their EIR/ECR at the chosen τ is out-of-fold.
- **Exploratory, labelled as such:** in-sample curves for the learned gates (fit and evaluated on all of dev) and breakdowns by pushback type.

## 4. Metrics
- **AUROC (good revision, among flips):** AUROC of the acceptance score for good vs bad revisions.
- **Area under the EIR–ECR curve:** sweep τ over the OOF acceptance scores from accept-none (EIR = ECR = 0) to accept-all (EIR_max, ECR_max). We report:
  - the raw area under ECR(EIR) on [0, EIR_max] (trapezoids);
  - the area normalised by EIR_max · ECR_max, where random = 0.5 and oracle = 1.

  Among flips, a good revision is the same as "a1 wrong", so the normalised area equals the AUROC (ties handled the same way). We report both because they were asked for; they are not two independent tests.
- **Random gate:** accept each flip independently with probability r, for r in a grid of 21 values from 0 to 1 and 100 seeds each. Plotted as a band (2.5–97.5 percentiles of EIR and ECR across seeds). Its expected curve is the straight line from (0, 0) to (EIR_max, ECR_max).
- **Oracle gate:** accept iff the revision is good. It gives EIR = 0 and ECR = ECR_max, the flip rate when a1 is wrong, which is the ceiling.
- **Uncertainty:** 95% percentile intervals from a bootstrap clustered by image (1,000 resamples, seed 0). OOF predictions are held fixed inside the bootstrap; models are not refitted. That understates the variance from fitting, and we state this as a limitation.

## 5. H2 decision rule (reliance adds information beyond confidence and shift)
- **Compare** learned gate B against learned gate A, out-of-fold, separately for each model.
- **Test statistic:** Δ = metric(B) − metric(A), for (i) the OOF AUROC (good revision) and (ii) the OOF area under the EIR–ECR curve.
- **Interval:** a **paired** image-clustered bootstrap, where the same resampled images are used for A and B. 1,000 resamples, seed 0, 95% percentile CI.
- **Per-model decision:**
  - CI entirely above 0: reliance adds information for this model.
  - CI includes 0: **no clear effect**.
  - CI entirely below 0: adding reliance hurts.
- **Overall:** "H2 supported" only if the CI is above 0 for **both** models. "Mixed" if it is above 0 for one. Otherwise "not supported", reported as a negative result.
- The AUROC difference is the primary result. The area difference is the same quantity rescaled (section 4), so it can't disagree in sign. It is reported as asked, not as extra evidence.

## 6. Net accuracy and the sensitivity to how often the user is right
- **Definitions:** π = P(user is right) = P(a1 wrong). Turn-1 accuracy is 1 − π, and final accuracy is (1 − π)(1 − EIR) + π · ECR.
- **Net accuracy change** = π · ECR − (1 − π) · EIR.
- **Main tuning and reporting:** at the empirical π of dev (the observed a1-wrong rate).
- **Sensitivity analysis:** π ∈ {0.1, 0.3, 0.5}. EIR and ECR are rates conditional on a1 correct or wrong, so changing π only reweights them.
  - For each π, thresholds are re-tuned on the training folds with the π-weighted objective, and evaluated out-of-fold.
  - Reported for every gate, plus always-accept, never-accept and the oracle.

## 7. Results split by turn-1 answer (yes vs no)
- For a1 = yes and a1 = no separately, report:
  - the signal AUROC for predicting a flip (conf1, reliance_blur, reliance_blank; **not** shift_to_claim);
  - the signal AUROC for predicting a good revision (all four signals);
  - the mean of each signal;

  each with clustered-bootstrap CIs.
- Purely descriptive; no decision rule.

## 8. H3 per POPE split (random, popular, adversarial)
- **Calibration of turn 1, per question (not per row):**
  - expected calibration error (10 equal-count bins of conf1);
  - mean conf1 on wrong answers;
  - share of wrong answers with conf1 > 0.9;
  - turn-1 accuracy;

  each with clustered-bootstrap CIs.
- **Gate performance per split:** the OOF AUROC (good revision) of each gate, computed on the rows of that split. The OOF predictions come from the pooled cross-validation in section 3.
- **Decision rule.** H3 is supported for a model only if both:
  1. the confidence gate degrades on adversarial: AUROC_conf(adversarial) − AUROC_conf(random) has a 95% CI entirely below 0; **and**
  2. reliance holds up better: the difference-in-differences [AUROC_blur(adv) − AUROC_blur(rand)] − [AUROC_conf(adv) − AUROC_conf(rand)] has a 95% CI entirely above 0.

  The bootstrap is paired, resampling images; every dev image appears in every POPE split. Anything else is **reported as a negative or inconclusive result**.
- **Power warning:** good revisions need a1 to be wrong. With about 13% turn-1 errors spread over three splits, each split has only a few dozen wrong questions, so these intervals will be wide.

## 9. Frozen gates for the test run
- After cross-validation, every gate is refit on **all of dev**:
  - thresholds at the empirical π and at each π in {0.1, 0.3, 0.5};
  - learned-gate coefficients, intercepts and standardisation means and SDs;
  - feature order and orientation.
- Saved to `results/frozen_gates.json` with the git commit.
- `run_final.py --final` will apply these unchanged to the test split, once. Nothing in Phase 5 reads the test split.

## 10. Multiple comparisons
No correction is applied. The primary tests are H2 (2 models × 1 comparison) and H3 (2 models). Everything else is descriptive and labelled so.
