# Numbers in `PROJECT_OVERVIEW.md` that need test-split counterparts

The dev numbers in the overview stay as they are: they are dev results. The report's **headline results** must come from the test files listed here. Every source is produced by the tagged run (`run_final.py --final` → `analyse_final.py`) or by `make_failure_gallery.py --partition test`.

All test tables are in `results/final_test/`, and every row carries `status = "FINAL test result …"`.

| Overview section (dev now) | Numbers / file names to replace | Test source (file → columns / rows) |
|---|---|---|
| §4.1 Data | "test: 300 images = 900 questions" (planned) → the count actually run | `results/signals_test_<model>.jsonl` (rows ÷ 12 = questions); `results/final_test/auroc.csv` is produced only if the run completed |
| §5.2 Baseline: turn-1 accuracy (overall and per POPE split), flip rate, flip rate when a1 is right / wrong, turn-2 accuracy, turn-2 yes/no mass, flip rate by pushback type and by split | every number in both §5.2 tables | **No final table yet.** Compute with `vlmsyc.metrics.flip_summary` on `results/baseline_test_<model>.jsonl` (needs a small descriptive script; see gap 1) |
| §5.2 "always-accept: EIR = flip rate when a1 right, ECR = flip rate when a1 wrong" | EIR_max, ECR_max | `results/final_test/references.csv` → `reference=always_accept`, `operating_point=native_pi`: `eir`, `ecr` |
| §5.3 Signal AUROCs (flip and good revision) and the by-answer split | the 4×2 AUROC table; the a1 = yes / no table | by answer: `results/final_test/by_a1.csv` → `a1`, `target`, `signal`, `auroc_raw_signal`, `ci_low`, `ci_high`. **Pooled (not split by answer): no final table** (gap 2) |
| §5.4 Gate AUROCs (OOF on dev) | per-gate AUROC and CIs, headline and secondary | `results/final_test/auroc.csv` → `gate`, `headline`, `auroc_good_revision`, `ci_low`, `ci_high`, `area`, `area_normalised` |
| §5.4 Net change at native π; always-accept / oracle net | "net ≈ 0 for gates", −0.383 / −0.349 (always), +0.072 / +0.060 (oracle) | gates: `results/final_test/gates.csv` → `operating_point=native_pi`: `eir`, `ecr`, `net_change`, `net_ci_low/high`, `beats_never_accept`, `net_minus_matched_random`, `vs_random_ci_low/high`; references: `references.csv` → `native_pi` |
| §5.4 figure | `figures/eir_ecr_dev.png` | `figures/final_test/eir_ecr.png` |
| §5.5 H2 table and verdict | A, B, B − A and CI per model; "H2 overall: mixed" | `results/final_test/h2.csv` → `metric=auroc_good_revision` (primary) and `area`: `gate_A`, `gate_B`, `difference_B_minus_A`, `ci_low`, `ci_high`, `decision`; overall label: `results/final_test/run_info.json` → `h2_overall`, `h2_per_model` |
| §5.6 Sensitivity table (π = 0.1 / 0.3 / 0.5) | always / best gate / oracle net per π | `results/final_test/gates.csv` → `operating_point ∈ {pi_0.1, pi_0.3, pi_0.5}`: `net_change`, CIs; `references.csv` → same `operating_point` values. Not a headline (π = 0.5 must not be headlined, Amendment A3) |
| §5.6 figure | `figures/sensitivity_pi_dev.png` | `figures/final_test/sensitivity_pi.png` |
| §5.7 H3: per-split gate AUROC | conf_gate / reliance_blur_gate AUROC per split | `results/final_test/h3.csv` → `row=split_auroc`: `gate`, `pope_split`, `value`, `ci_low`, `ci_high` |
| §5.7 H3: decision quantities and verdict | adv − random (conf), difference-in-differences, per-model verdicts, overall | `h3.csv` → `row=conf_gate_adv_minus_random`, `row=did_blur_minus_conf`, `row=verdict`; overall: `run_info.json` → `h3_overall`, `h3_per_model` |
| §5.7 H3: calibration (turn-1 accuracy per split, mean conf1 when wrong, share wrong with conf1 > 0.9) | the three calibration rows | **No final table** (gap 3) |
| §5.7 figure | `figures/reliability_by_split_dev.png` | **No final figure** (gap 3) |
| §5.9 Grounding / re-look (not no-suffix: dev only) | flip rate, EIR, ECR, turn-2 accuracy, Δ vs base with CIs, mass | `results/final_test/mitigations.csv` → `condition ∈ {grounding, relook}`: `flip_rate`, `eir`, `ecr`, `turn2_acc`, `mean_mass_t2`, `*_minus_base` and CIs, `net_change_native_pi`, `net_change_pi_*` |
| §5.10 Runtimes for job A / B | "about 1.3 h / 2 h" (estimates) | `sacct -j 73424,73425 --format=Elapsed` on TC1 |
| Failure gallery (§5.11, new) | gallery figure and counts | `figures/failure_gallery_test_<model>.png`, `results/failure_gallery_test.csv`, `results/failure_counts_test.csv` (from `make_failure_gallery.py --partition test`) |

## Gaps: test numbers the tagged analysis does not produce
1. **Baseline flip summary on test** (§5.2). It is descriptive and needs no decision rule. A small new script reading `results/baseline_test_<model>.jsonl` with the existing `vlmsyc.metrics.flip_summary` would produce it.
2. **Pooled signal AUROC table on test** (§5.3). The tagged analysis only writes the by-answer split. `scripts/run_signals.py --join-only` computes the pooled table, but only for dev file names.
3. **Turn-1 calibration per split on test** (§5.7, three rows and the reliability figure). The tagged `analyse_final.py` has no calibration step; `fit_gates_dev.py` computes it, for dev only.

None of these changes a threshold, gate or decision rule. They are descriptive additions over the saved test outputs. Whether to add them, and how to disclose them in the report, is a team decision. Note C covers *fixing* the analysis script, not adding new descriptive analyses.
