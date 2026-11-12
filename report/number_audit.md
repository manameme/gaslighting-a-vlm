# Number audit of the report draft

Checked 363 items. Status counts: {'OK': 348, 'NOTE': 5, 'MISMATCH': 2, 'RESOLVED': 1, 'UNTRACEABLE to results/': 7}

| Section | Item | As written | Exact value in source | Source | Status |
|---|---|---|---|---|---|
| Abstract | Qwen turn-1 accuracy | 0.872 | 0.872222 | describe_test/flip_summary.csv row 2, col `turn1_acc` | OK |
| Abstract | Qwen turn-2 accuracy (accept all) | 0.489 | 0.488796 | describe_test/flip_summary.csv row 2, col `turn2_acc` | OK |
| Abstract | LLaVA turn-1 accuracy | 0.868 | 0.867778 | describe_test/flip_summary.csv row 10, col `turn1_acc` | OK |
| Abstract | LLaVA turn-2 accuracy | 0.515 | 0.515093 | describe_test/flip_summary.csv row 10, col `turn2_acc` | OK |
| Abstract | H2 Qwen B-A AUROC | +0.023 | 0.022804 | final_test/h2.csv row 2, col `difference_B_minus_A` | OK |
| Abstract | H2 LLaVA B-A AUROC | -0.004 | -0.003921 | final_test/h2.csv row 4, col `difference_B_minus_A` | OK |
| Abstract | 'about 13%' = native pi (qwen2vl_2b) | 0.13 | 0.127778 | final_test/references.csv row 3, col `pi` | OK |
| Abstract | 'about 13%' = native pi (llava15_7b_fp16) | 0.13 | 0.132222 | final_test/references.csv row 15, col `pi` | OK |
| Abstract | 'no gate gains more than about one point' at native pi (qwen2vl_2b): best gate | ≤ ~0.01 | 0.010000 (reliance_blur_gate) | final_test/gates.csv row 6, col `net_change` (max over all 6 gates) | OK |
| Abstract | 'no gate gains more than about one point' at native pi (llava15_7b_fp16): best gate | ≤ ~0.01 | -0.001296 (learned_B_conf_shift_reliance) | final_test/gates.csv row 46, col `net_change` (max over all 6 gates) | OK |
| Abstract | 'Gates help when users are right ≥ ~30%': all headline gates at pi=0.3 have CI > 0 | — | min headline net_ci_low = 0.009649 | final_test/gates.csv rows with operating_point=pi_0.3, col `net_ci_low` | OK |
| Abstract | … but not every secondary gate | — | llava15_7b_fp16 reliance_blur_gate CI low -0.0014 (row 32) | final_test/gates.csv | NOTE: 'gates' should read 'headline gates' |
| Abstract | pi = 0.5 oracle share absent from Abstract | — | not present in Abstract text | report text | OK (confirmed) |
| Introduction | H1 supported (qwen2vl_2b) | supported | supported | describe_test/h1.csv, col `h1_verdict` | OK |
| Introduction | H1 supported (llava15_7b_fp16) | supported | supported | describe_test/h1.csv, col `h1_verdict` | OK |
| Introduction | H2 mixed | mixed | mixed | final_test/run_info.json `h2_overall` | OK |
| Introduction | H3 not supported | not supported | not supported | final_test/run_info.json `h3_overall` | OK |
| Methods | 500 images / 200 dev / 300 test | 500 / 200 / 300 | 500 / 200 / 300 | data/splits.json `n_images`, `images` | OK |
| Methods | 1,500 questions / 600 dev / 900 test | 1,500 / 600 / 900 | 1500 / 600 / 900 | data/splits.json `questions` | OK |
| Methods | No image in both partitions | none | see file | results/splits_summary.csv | OK |
| Methods | Pilot: 100 questions | 100 | 100 / 100 | results/pilot.csv `n_questions` (qwen2vl_2b instr, llava15_7b_fp16 instr) | OK |
| Methods | Pilot flip rates within ~20–70% | 20%–70% | Qwen 0.4700, LLaVA fp16 0.4875 | results/pilot.csv `flip_rate` (variant instr) | OK |
| Methods | Blur sigma 3% of longer side | 3% | blur_sigma_frac: 0.03 | configs/experiment.yaml `signals.blur_sigma_frac` | OK |
| Methods | 4 styles x 3 wordings = 12 pushbacks | 12 | 4 types x 3 paraphrases | configs/pushback.yaml `pushback` | OK |
| Methods | Native pi Qwen | 0.128 | 0.127778 | final_test/references.csv row 3, col `pi` | OK |
| Methods | Native pi LLaVA | 0.132 | 0.132222 | final_test/references.csv row 15, col `pi` | OK |
| Methods | 1,000 bootstrap resamples | 1,000 | 1000 | final_test/run_info.json `n_boot` | OK |
| Methods | Random gate averaged over 100 seeds | 100 | n_seeds=100 | scripts/analyse_final.py (random_gate_band call) | OK |
| Methods | Four of five folds | 5 folds | K_FOLDS = 5 | scripts/fit_gates_dev.py | OK |
| Methods | Frozen-gates sha256 | 71979e78…cebe86 | 71979e78b8cd69a6281e3c04ef0c3502b1b62ff20473196ffd2caab39ccebe86 | results/frozen_gates.json (sha256) | OK |
| Methods | Tag commit | 8269e4ff…5371189d | 8269e4ff3f08a459246c94b27230733a5371189d | git rev-parse pre-final^{commit} | OK |
| Methods | 'Two changes … recorded in the plan (Amendment A): shift gate reversed; pushback type added' | Amendment A | shift orientation: plan §2 (c18dfe8); pushback type: Amendment A1 (c043521) | results/analysis_plan.md | MISMATCH (attribution): only pushback type is Amendment A; the shift reversal is in §2 of the original plan |
| Methods | Tesla V100 32 GB | V100 (32 GB) | Tesla V100-PCIE-32GB, 32768 MiB | results/logs_tc1/output_finalfA_73424.out | OK |
| Results: behaviour / Table 1 | Table 1 qwen2vl_2b turn1_acc | 0.872 | 0.872222 | describe_test/flip_summary.csv row 2, col `turn1_acc` | OK |
| Results: behaviour / Table 1 | Table 1 qwen2vl_2b turn2_acc | 0.489 | 0.488796 | describe_test/flip_summary.csv row 2, col `turn2_acc` | OK |
| Results: behaviour / Table 1 | Table 1 qwen2vl_2b flip_rate | 0.532 | 0.531944 | describe_test/flip_summary.csv row 2, col `flip_rate` | OK |
| Results: behaviour / Table 1 | Table 1 qwen2vl_2b flip_rate_a1_correct | 0.525 | 0.524735 | describe_test/flip_summary.csv row 2, col `flip_rate_a1_correct` | OK |
| Results: behaviour / Table 1 | Table 1 qwen2vl_2b flip_rate_a1_wrong | 0.581 | 0.581159 | describe_test/flip_summary.csv row 2, col `flip_rate_a1_wrong` | OK |
| Results: behaviour / Table 1 | Table 1 llava15_7b_fp16 turn1_acc | 0.868 | 0.867778 | describe_test/flip_summary.csv row 10, col `turn1_acc` | OK |
| Results: behaviour / Table 1 | Table 1 llava15_7b_fp16 turn2_acc | 0.515 | 0.515093 | describe_test/flip_summary.csv row 10, col `turn2_acc` | OK |
| Results: behaviour / Table 1 | Table 1 llava15_7b_fp16 flip_rate | 0.474 | 0.473981 | describe_test/flip_summary.csv row 10, col `flip_rate` | OK |
| Results: behaviour / Table 1 | Table 1 llava15_7b_fp16 flip_rate_a1_correct | 0.476 | 0.476312 | describe_test/flip_summary.csv row 10, col `flip_rate_a1_correct` | OK |
| Results: behaviour / Table 1 | Table 1 llava15_7b_fp16 flip_rate_a1_wrong | 0.459 | 0.458683 | describe_test/flip_summary.csv row 10, col `flip_rate_a1_wrong` | OK |
| Results: behaviour / Table 1 | Qwen flip 'Are you sure?' | 0.161 | 0.161111 | describe_test/flip_summary.csv row 5, col `flip_rate` | OK |
| Results: behaviour / Table 1 | Qwen flip assertive | 0.836 | 0.835556 | describe_test/flip_summary.csv row 3, col `flip_rate` | OK |
| Results: behaviour / Table 1 | LLaVA flip neutral | 0.374 | 0.374444 | describe_test/flip_summary.csv row 13, col `flip_rate` | OK |
| Results: behaviour / Table 1 | LLaVA flip assertive | 0.586 | 0.586296 | describe_test/flip_summary.csv row 11, col `flip_rate` | OK |
| Results: behaviour / Table 1 | '10,800 question-pushback rows per model' | 10,800 | 10800 / 10800 | failure_counts_test.csv `n_rows` (subset all rows) | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b conf1 a1=yes auroc | 0.473 | 0.473152 | describe_test/flip_auroc_by_a1.csv row 2, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b conf1 a1=yes ci_low | 0.466 | 0.466098 | describe_test/flip_auroc_by_a1.csv row 2, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b conf1 a1=yes ci_high | 0.480 | 0.480427 | describe_test/flip_auroc_by_a1.csv row 2, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b conf1 a1=no auroc | 0.313 | 0.313158 | describe_test/flip_auroc_by_a1.csv row 5, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b conf1 a1=no ci_low | 0.292 | 0.292085 | describe_test/flip_auroc_by_a1.csv row 5, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b conf1 a1=no ci_high | 0.333 | 0.332949 | describe_test/flip_auroc_by_a1.csv row 5, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blur a1=yes auroc | 0.488 | 0.488160 | describe_test/flip_auroc_by_a1.csv row 3, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blur a1=yes ci_low | 0.482 | 0.481642 | describe_test/flip_auroc_by_a1.csv row 3, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blur a1=yes ci_high | 0.494 | 0.494235 | describe_test/flip_auroc_by_a1.csv row 3, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blur a1=no auroc | 0.455 | 0.455449 | describe_test/flip_auroc_by_a1.csv row 6, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blur a1=no ci_low | 0.429 | 0.428760 | describe_test/flip_auroc_by_a1.csv row 6, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blur a1=no ci_high | 0.484 | 0.483801 | describe_test/flip_auroc_by_a1.csv row 6, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blank a1=yes auroc | 0.474 | 0.474053 | describe_test/flip_auroc_by_a1.csv row 4, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blank a1=yes ci_low | 0.467 | 0.466759 | describe_test/flip_auroc_by_a1.csv row 4, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blank a1=yes ci_high | 0.481 | 0.481297 | describe_test/flip_auroc_by_a1.csv row 4, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blank a1=no auroc | 0.355 | 0.355169 | describe_test/flip_auroc_by_a1.csv row 7, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blank a1=no ci_low | 0.333 | 0.332970 | describe_test/flip_auroc_by_a1.csv row 7, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 qwen2vl_2b reliance_blank a1=no ci_high | 0.379 | 0.378573 | describe_test/flip_auroc_by_a1.csv row 7, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 conf1 a1=yes auroc | 0.485 | 0.484597 | describe_test/flip_auroc_by_a1.csv row 8, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 conf1 a1=yes ci_low | 0.480 | 0.479952 | describe_test/flip_auroc_by_a1.csv row 8, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 conf1 a1=yes ci_high | 0.490 | 0.489555 | describe_test/flip_auroc_by_a1.csv row 8, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 conf1 a1=no auroc | 0.362 | 0.362079 | describe_test/flip_auroc_by_a1.csv row 11, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 conf1 a1=no ci_low | 0.351 | 0.351097 | describe_test/flip_auroc_by_a1.csv row 11, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 conf1 a1=no ci_high | 0.374 | 0.374199 | describe_test/flip_auroc_by_a1.csv row 11, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blur a1=yes auroc | 0.492 | 0.491910 | describe_test/flip_auroc_by_a1.csv row 9, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blur a1=yes ci_low | 0.489 | 0.488840 | describe_test/flip_auroc_by_a1.csv row 9, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blur a1=yes ci_high | 0.495 | 0.494830 | describe_test/flip_auroc_by_a1.csv row 9, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blur a1=no auroc | 0.416 | 0.415630 | describe_test/flip_auroc_by_a1.csv row 12, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blur a1=no ci_low | 0.396 | 0.396089 | describe_test/flip_auroc_by_a1.csv row 12, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blur a1=no ci_high | 0.437 | 0.437078 | describe_test/flip_auroc_by_a1.csv row 12, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blank a1=yes auroc | 0.485 | 0.484581 | describe_test/flip_auroc_by_a1.csv row 10, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blank a1=yes ci_low | 0.480 | 0.479917 | describe_test/flip_auroc_by_a1.csv row 10, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blank a1=yes ci_high | 0.490 | 0.489537 | describe_test/flip_auroc_by_a1.csv row 10, col `ci_high` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blank a1=no auroc | 0.362 | 0.362079 | describe_test/flip_auroc_by_a1.csv row 13, col `auroc` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blank a1=no ci_low | 0.351 | 0.351117 | describe_test/flip_auroc_by_a1.csv row 13, col `ci_low` | OK |
| Results: H1 / Table 2 | Table 2 llava15_7b_fp16 reliance_blank a1=no ci_high | 0.374 | 0.374211 | describe_test/flip_auroc_by_a1.csv row 13, col `ci_high` | OK |
| Results: H1 / Table 2 | 'yes' answers 0.47 to 0.49 (conf1) | 0.47–0.49 | 0.4732 (Qwen), 0.4846 (LLaVA) | describe_test/flip_auroc_by_a1.csv col `auroc` | OK |
| Results: H1 / Table 2 | 'no' answers 0.31 to 0.36 (conf1) | 0.31–0.36 | 0.3132 (Qwen), 0.3621 (LLaVA) | describe_test/flip_auroc_by_a1.csv col `auroc` | OK |
| Results: H1 / Table 2 | Pooled blur Qwen | 0.724 | 0.723907 | describe_test/flip_auroc_pooled_context.csv row 3, col `auroc` | OK |
| Results: H1 / Table 2 | Pooled blur LLaVA | 0.710 | 0.710080 | describe_test/flip_auroc_pooled_context.csv row 15, col `auroc` | OK |
| Results: H1 / Table 2 | Blur within answer 0.416 to 0.492 | 0.416–0.492 | min 0.4157 (LLaVA no), max 0.4919 (LLaVA yes) | describe_test/flip_auroc_by_a1.csv col `auroc`, signal reliance_blur | OK |
| Results: H1 / Table 2 | 'Yes answers flip more … higher blur reliance in the development data' | dev | test also: Qwen yes flips 3824/4524 vs no 1921/6276; blur mean yes 0.533 vs no 0.106 (LLaVA 0.584 vs -0.020) | describe_test/flip_auroc_by_a1.csv cols `n_positive`, `n_rows`, `mean_signal` | NOTE: the same holds on test; can cite test |
| Results: H1 / Table 2 | [Check] LLaVA blank AUROC = conf1 AUROC | — | p(yes/blank) 0.010–0.056, SD 0.0056; corr(conf1, reliance_blank) 0.999 | describe_test/blank_check.csv | RESOLVED: not a bug |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b conf_gate auroc_good_revision | 0.801 | 0.801391 | final_test/auroc.csv row 2, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b conf_gate ci_low | 0.760 | 0.760162 | final_test/auroc.csv row 2, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b conf_gate ci_high | 0.838 | 0.837846 | final_test/auroc.csv row 2, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 conf_gate auroc_good_revision | 0.827 | 0.827052 | final_test/auroc.csv row 8, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 conf_gate ci_low | 0.783 | 0.783056 | final_test/auroc.csv row 8, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 conf_gate ci_high | 0.865 | 0.865194 | final_test/auroc.csv row 8, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b shift_gate auroc_good_revision | 0.824 | 0.823798 | final_test/auroc.csv row 5, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b shift_gate ci_low | 0.790 | 0.789665 | final_test/auroc.csv row 5, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b shift_gate ci_high | 0.856 | 0.856194 | final_test/auroc.csv row 5, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 shift_gate auroc_good_revision | 0.826 | 0.826330 | final_test/auroc.csv row 11, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 shift_gate ci_low | 0.787 | 0.787096 | final_test/auroc.csv row 11, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 shift_gate ci_high | 0.862 | 0.861837 | final_test/auroc.csv row 11, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b reliance_blur_gate auroc_good_revision | 0.785 | 0.784608 | final_test/auroc.csv row 3, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b reliance_blur_gate ci_low | 0.732 | 0.731533 | final_test/auroc.csv row 3, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b reliance_blur_gate ci_high | 0.837 | 0.836621 | final_test/auroc.csv row 3, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 reliance_blur_gate auroc_good_revision | 0.703 | 0.702686 | final_test/auroc.csv row 9, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 reliance_blur_gate ci_low | 0.647 | 0.646580 | final_test/auroc.csv row 9, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 reliance_blur_gate ci_high | 0.753 | 0.753364 | final_test/auroc.csv row 9, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b reliance_blank_gate auroc_good_revision | 0.835 | 0.834631 | final_test/auroc.csv row 4, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b reliance_blank_gate ci_low | 0.795 | 0.794875 | final_test/auroc.csv row 4, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b reliance_blank_gate ci_high | 0.872 | 0.871892 | final_test/auroc.csv row 4, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 reliance_blank_gate auroc_good_revision | 0.786 | 0.786423 | final_test/auroc.csv row 10, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 reliance_blank_gate ci_low | 0.742 | 0.742158 | final_test/auroc.csv row 10, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 reliance_blank_gate ci_high | 0.829 | 0.829252 | final_test/auroc.csv row 10, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b learned_A_conf_shift auroc_good_revision | 0.845 | 0.845070 | final_test/auroc.csv row 6, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b learned_A_conf_shift ci_low | 0.812 | 0.812196 | final_test/auroc.csv row 6, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b learned_A_conf_shift ci_high | 0.878 | 0.878310 | final_test/auroc.csv row 6, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 learned_A_conf_shift auroc_good_revision | 0.837 | 0.837263 | final_test/auroc.csv row 12, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 learned_A_conf_shift ci_low | 0.796 | 0.796449 | final_test/auroc.csv row 12, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 learned_A_conf_shift ci_high | 0.873 | 0.873246 | final_test/auroc.csv row 12, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b learned_B_conf_shift_reliance auroc_good_revision | 0.868 | 0.867875 | final_test/auroc.csv row 7, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b learned_B_conf_shift_reliance ci_low | 0.835 | 0.834503 | final_test/auroc.csv row 7, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 qwen2vl_2b learned_B_conf_shift_reliance ci_high | 0.902 | 0.902443 | final_test/auroc.csv row 7, col `ci_high` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 learned_B_conf_shift_reliance auroc_good_revision | 0.833 | 0.833342 | final_test/auroc.csv row 13, col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 learned_B_conf_shift_reliance ci_low | 0.791 | 0.790978 | final_test/auroc.csv row 13, col `ci_low` | OK |
| Results: H2 / Table 3 | Table 3 llava15_7b_fp16 learned_B_conf_shift_reliance ci_high | 0.870 | 0.870292 | final_test/auroc.csv row 13, col `ci_high` | OK |
| Results: H2 / Table 3 | 'AUROC 0.70 to 0.87' | 0.70–0.87 | min 0.7029 (LLaVA blur), max 0.8679 (Qwen B) | final_test/auroc.csv col `auroc_good_revision` | OK |
| Results: H2 / Table 3 | H2 prose qwen2vl_2b gate_B | 0.868 | 0.867875 | final_test/h2.csv row 2, col `gate_B` | OK |
| Results: H2 / Table 3 | H2 prose qwen2vl_2b gate_A | 0.845 | 0.845070 | final_test/h2.csv row 2, col `gate_A` | OK |
| Results: H2 / Table 3 | H2 prose qwen2vl_2b difference_B_minus_A | +0.023 | 0.022804 | final_test/h2.csv row 2, col `difference_B_minus_A` | OK |
| Results: H2 / Table 3 | H2 prose qwen2vl_2b ci_low | +0.008 | 0.007793 | final_test/h2.csv row 2, col `ci_low` | OK |
| Results: H2 / Table 3 | H2 prose qwen2vl_2b ci_high | +0.040 | 0.039677 | final_test/h2.csv row 2, col `ci_high` | OK |
| Results: H2 / Table 3 | H2 prose llava15_7b_fp16 gate_B | 0.833 | 0.833342 | final_test/h2.csv row 4, col `gate_B` | OK |
| Results: H2 / Table 3 | H2 prose llava15_7b_fp16 gate_A | 0.837 | 0.837263 | final_test/h2.csv row 4, col `gate_A` | OK |
| Results: H2 / Table 3 | H2 prose llava15_7b_fp16 difference_B_minus_A | -0.004 | -0.003921 | final_test/h2.csv row 4, col `difference_B_minus_A` | OK |
| Results: H2 / Table 3 | H2 prose llava15_7b_fp16 ci_low | -0.013 | -0.013250 | final_test/h2.csv row 4, col `ci_low` | OK |
| Results: H2 / Table 3 | H2 prose llava15_7b_fp16 ci_high | +0.006 | 0.005515 | final_test/h2.csv row 4, col `ci_high` | OK |
| Results: H2 / Table 3 | H2 area qwen2vl_2b difference_B_minus_A | +0.0070 | 0.006954 | final_test/h2.csv row 3, col `difference_B_minus_A` | OK |
| Results: H2 / Table 3 | H2 area qwen2vl_2b ci_low | +0.0023 | 0.002325 | final_test/h2.csv row 3, col `ci_low` | OK |
| Results: H2 / Table 3 | H2 area qwen2vl_2b ci_high | +0.0121 | 0.012083 | final_test/h2.csv row 3, col `ci_high` | OK |
| Results: H2 / Table 3 | H2 area llava15_7b_fp16 difference_B_minus_A | -0.0009 | -0.000857 | final_test/h2.csv row 5, col `difference_B_minus_A` | OK |
| Results: H2 / Table 3 | H2 area llava15_7b_fp16 ci_low | -0.0030 | -0.003037 | final_test/h2.csv row 5, col `ci_low` | OK |
| Results: H2 / Table 3 | H2 area llava15_7b_fp16 ci_high | +0.0011 | 0.001129 | final_test/h2.csv row 5, col `ci_high` | OK |
| Results: sensitivity / Table 4 | Headline gates accept 1% to 4% of flips | 1%–4% | 1.41% to 4.07% | final_test/gates.csv col `accept_rate_among_flips` (native_pi, headline) | OK (rounded) |
| Results: sensitivity / Table 4 | Oracle accepts 13% to 14% | 13%–14% | 12.80% to 13.96% | final_test/references.csv col `accept_rate_among_flips` (oracle, native_pi) | OK (rounded) |
| Results: sensitivity / Table 4 | EIR at or below 0.015 | ≤ 0.015 | max 0.014544 (qwen2vl_2b conf_gate, row 2) | final_test/gates.csv col `eir` | OK (rounded) |
| Results: sensitivity / Table 4 | ECR 0.007 to 0.104 | 0.007–0.104 | 0.007003 (row 26) to 0.103623 (row 22) | final_test/gates.csv col `ecr` | OK (rounded) |
| Results: sensitivity / Table 4 | Qwen B native net_change | +0.009 | 0.009074 | final_test/gates.csv row 22, col `net_change` | OK |
| Results: sensitivity / Table 4 | Qwen B native net_ci_low | +0.004 | 0.003730 | final_test/gates.csv row 22, col `net_ci_low` | OK |
| Results: sensitivity / Table 4 | Qwen B native net_ci_high | +0.015 | 0.015056 | final_test/gates.csv row 22, col `net_ci_high` | OK |
| Results: sensitivity / Table 4 | LLaVA conf native net_change | -0.005 | -0.004815 | final_test/gates.csv row 26, col `net_change` | OK |
| Results: sensitivity / Table 4 | LLaVA conf native net_ci_low | -0.010 | -0.009670 | final_test/gates.csv row 26, col `net_ci_low` | OK |
| Results: sensitivity / Table 4 | LLaVA conf native net_ci_high | -0.001 | -0.000583 | final_test/gates.csv row 26, col `net_ci_high` | OK |
| Results: sensitivity / Table 4 | 'Only Qwen's learned gate B has an interval above zero' (native pi) | only Qwen B | qwen2vl_2b reliance_blur_gate (CI low 0.0033, row 6); qwen2vl_2b learned_B_conf_shift_reliance (CI low 0.0037, row 22) | final_test/gates.csv col `net_ci_low` | NOTE: true for headline gates; secondary Qwen reliance_blur_gate also > 0 — say 'of the headline gates' |
| Results: sensitivity / Table 4 | 'LLaVA's confidence gate is slightly but reliably negative' (native) | one gate | llava15_7b_fp16 conf_gate (CI high -0.0006, row 26); llava15_7b_fp16 shift_gate (CI high -0.0003, row 38) | final_test/gates.csv col `net_ci_high` | NOTE: among headline gates only LLaVA conf; secondary LLaVA shift_gate also < 0 |
| Results: sensitivity / Table 4 | pi=0.3 headline range qwen2vl_2b | +0.050 to +0.059 | 0.049733 to 0.059028; all CI low > 0: True | final_test/gates.csv col `net_change`, `net_ci_low` (pi_0.3) | OK |
| Results: sensitivity / Table 4 | pi=0.3 headline range llava15_7b_fp16 | +0.028 to +0.037 | 0.027924 to 0.037484; all CI low > 0: True | final_test/gates.csv col `net_change`, `net_ci_low` (pi_0.3) | OK |
| Results: sensitivity / Table 4 | pi=0.5: 0.46 to 0.59 of oracle's gain | 0.46–0.59 | 0.4654 to 0.5884 | final_test/gates.csv `net_change` (pi_0.5) / final_test/references.csv oracle `net_change` (pi_0.5) | OK (rounded) |
| Results: sensitivity / Table 4 | Qwen always-accept pi=0.5 | +0.028 | 0.028212 | final_test/references.csv row 11, col `net_change` | OK |
| Results: sensitivity / Table 4 | Qwen B pi=0.5 | +0.171 | 0.170989 | final_test/gates.csv row 25, col `net_change` | OK |
| Results: sensitivity / Table 4 | Always-accept worse than never except Qwen pi=0.5 | only Qwen pi=0.5 | qwen2vl_2b pi_0.5 +0.0282 | final_test/references.csv col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b always_accept native_pi | -0.383 | -0.383426 | final_test/references.csv row 2, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b always_accept pi_0.1 | -0.414 | -0.414145 | final_test/references.csv row 5, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b always_accept pi_0.3 | -0.193 | -0.192966 | final_test/references.csv row 8, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b always_accept pi_0.5 | +0.028 | 0.028212 | final_test/references.csv row 11, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b conf_gate native_pi | -0.004 | -0.003704 | final_test/gates.csv row 2, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b conf_gate pi_0.1 | -0.006 | -0.006060 | final_test/gates.csv row 3, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b conf_gate pi_0.3 | +0.050 | 0.049733 | final_test/gates.csv row 4, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b conf_gate pi_0.5 | +0.135 | 0.135244 | final_test/gates.csv row 5, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_A_conf_shift native_pi | +0.003 | 0.002500 | final_test/gates.csv row 18, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_A_conf_shift pi_0.1 | +0.001 | 0.000862 | final_test/gates.csv row 19, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_A_conf_shift pi_0.3 | +0.051 | 0.051051 | final_test/gates.csv row 20, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_A_conf_shift pi_0.5 | +0.161 | 0.161112 | final_test/gates.csv row 21, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_B_conf_shift_reliance native_pi | +0.009 | 0.009074 | final_test/gates.csv row 22, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_B_conf_shift_reliance pi_0.1 | +0.006 | 0.005697 | final_test/gates.csv row 23, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_B_conf_shift_reliance pi_0.3 | +0.059 | 0.059028 | final_test/gates.csv row 24, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b learned_B_conf_shift_reliance pi_0.5 | +0.171 | 0.170989 | final_test/gates.csv row 25, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b oracle native_pi | +0.074 | 0.074259 | final_test/references.csv row 4, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b oracle pi_0.1 | +0.058 | 0.058116 | final_test/references.csv row 7, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b oracle pi_0.3 | +0.174 | 0.174348 | final_test/references.csv row 10, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 qwen2vl_2b oracle pi_0.5 | +0.291 | 0.290580 | final_test/references.csv row 13, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 always_accept native_pi | -0.353 | -0.352685 | final_test/references.csv row 14, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 always_accept pi_0.1 | -0.383 | -0.382813 | final_test/references.csv row 17, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 always_accept pi_0.3 | -0.196 | -0.195814 | final_test/references.csv row 20, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 always_accept pi_0.5 | -0.009 | -0.008814 | final_test/references.csv row 23, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 conf_gate native_pi | -0.005 | -0.004815 | final_test/gates.csv row 26, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 conf_gate pi_0.1 | -0.005 | -0.005254 | final_test/gates.csv row 27, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 conf_gate pi_0.3 | +0.028 | 0.027924 | final_test/gates.csv row 28, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 conf_gate pi_0.5 | +0.114 | 0.114416 | final_test/gates.csv row 29, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_A_conf_shift native_pi | -0.002 | -0.002222 | final_test/gates.csv row 42, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_A_conf_shift pi_0.1 | -0.001 | -0.001360 | final_test/gates.csv row 43, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_A_conf_shift pi_0.3 | +0.037 | 0.037484 | final_test/gates.csv row 44, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_A_conf_shift pi_0.5 | +0.127 | 0.126970 | final_test/gates.csv row 45, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_B_conf_shift_reliance native_pi | -0.001 | -0.001296 | final_test/gates.csv row 46, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_B_conf_shift_reliance pi_0.1 | 0.000 | -0.000006 | final_test/gates.csv row 47, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_B_conf_shift_reliance pi_0.3 | +0.035 | 0.034967 | final_test/gates.csv row 48, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 learned_B_conf_shift_reliance pi_0.5 | +0.126 | 0.125919 | final_test/gates.csv row 49, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 oracle native_pi | +0.061 | 0.060648 | final_test/references.csv row 16, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 oracle pi_0.1 | +0.046 | 0.045868 | final_test/references.csv row 19, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 oracle pi_0.3 | +0.138 | 0.137605 | final_test/references.csv row 22, col `net_change` | OK |
| Results: sensitivity / Table 4 | Table 4 llava15_7b_fp16 oracle pi_0.5 | +0.229 | 0.229342 | final_test/references.csv row 25, col `net_change` | OK |
| Results: H3 / Table 5 | H3 qwen2vl_2b conf_gate_adv_minus_random value | -0.083 | -0.083410 | final_test/h3.csv row 14, col `value` | OK |
| Results: H3 / Table 5 | H3 qwen2vl_2b conf_gate_adv_minus_random ci_low | -0.170 | -0.170178 | final_test/h3.csv row 14, col `ci_low` | OK |
| Results: H3 / Table 5 | H3 qwen2vl_2b conf_gate_adv_minus_random ci_high | +0.005 | 0.004627 | final_test/h3.csv row 14, col `ci_high` | OK |
| Results: H3 / Table 5 | H3 qwen2vl_2b did_blur_minus_conf value | -0.041 | -0.040502 | final_test/h3.csv row 15, col `value` | OK |
| Results: H3 / Table 5 | H3 qwen2vl_2b did_blur_minus_conf ci_low | -0.183 | -0.183407 | final_test/h3.csv row 15, col `ci_low` | OK |
| Results: H3 / Table 5 | H3 qwen2vl_2b did_blur_minus_conf ci_high | +0.095 | 0.095424 | final_test/h3.csv row 15, col `ci_high` | OK |
| Results: H3 / Table 5 | H3 llava15_7b_fp16 conf_gate_adv_minus_random value | -0.074 | -0.073886 | final_test/h3.csv row 29, col `value` | OK |
| Results: H3 / Table 5 | H3 llava15_7b_fp16 conf_gate_adv_minus_random ci_low | -0.189 | -0.188697 | final_test/h3.csv row 29, col `ci_low` | OK |
| Results: H3 / Table 5 | H3 llava15_7b_fp16 conf_gate_adv_minus_random ci_high | +0.033 | 0.032782 | final_test/h3.csv row 29, col `ci_high` | OK |
| Results: H3 / Table 5 | H3 llava15_7b_fp16 did_blur_minus_conf value | +0.016 | 0.015549 | final_test/h3.csv row 30, col `value` | OK |
| Results: H3 / Table 5 | H3 llava15_7b_fp16 did_blur_minus_conf ci_low | -0.175 | -0.175038 | final_test/h3.csv row 30, col `ci_low` | OK |
| Results: H3 / Table 5 | H3 llava15_7b_fp16 did_blur_minus_conf ci_high | +0.207 | 0.207280 | final_test/h3.csv row 30, col `ci_high` | OK |
| Results: H3 / Table 5 | Dev 44% (Qwen adversarial, conf1 > 0.9) | 44% | 0.4444444444444444 | results/h3_calibration_dev.csv row 4, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | Test 23% (Qwen adversarial) | 0.23 | 0.230769 | describe_test/calibration.csv row 4, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | 26 to 52 wrong answers per split and model | 26–52 | 26 to 52 | describe_test/calibration.csv col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b random n_wrong | 27 | 27.000000 | describe_test/calibration.csv row 2, col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b random turn1_accuracy | 0.910 | 0.910000 | describe_test/calibration.csv row 2, col `turn1_accuracy` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b random mean_conf1_when_wrong | 0.786 | 0.785503 | describe_test/calibration.csv row 2, col `mean_conf1_when_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b random share_wrong_with_conf1_over_0.9 | 0.222 | 0.222222 | describe_test/calibration.csv row 2, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b popular n_wrong | 36 | 36.000000 | describe_test/calibration.csv row 3, col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b popular turn1_accuracy | 0.880 | 0.880000 | describe_test/calibration.csv row 3, col `turn1_accuracy` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b popular mean_conf1_when_wrong | 0.798 | 0.798334 | describe_test/calibration.csv row 3, col `mean_conf1_when_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b popular share_wrong_with_conf1_over_0.9 | 0.444 | 0.444444 | describe_test/calibration.csv row 3, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b adversarial n_wrong | 52 | 52.000000 | describe_test/calibration.csv row 4, col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b adversarial turn1_accuracy | 0.827 | 0.826667 | describe_test/calibration.csv row 4, col `turn1_accuracy` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b adversarial mean_conf1_when_wrong | 0.762 | 0.761912 | describe_test/calibration.csv row 4, col `mean_conf1_when_wrong` | OK |
| Results: H3 / Table 5 | Table 5 qwen2vl_2b adversarial share_wrong_with_conf1_over_0.9 | 0.231 | 0.230769 | describe_test/calibration.csv row 4, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 random n_wrong | 26 | 26.000000 | describe_test/calibration.csv row 5, col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 random turn1_accuracy | 0.913 | 0.913333 | describe_test/calibration.csv row 5, col `turn1_accuracy` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 random mean_conf1_when_wrong | 0.757 | 0.756730 | describe_test/calibration.csv row 5, col `mean_conf1_when_wrong` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 random share_wrong_with_conf1_over_0.9 | 0.231 | 0.230769 | describe_test/calibration.csv row 5, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 popular n_wrong | 47 | 47.000000 | describe_test/calibration.csv row 6, col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 popular turn1_accuracy | 0.843 | 0.843333 | describe_test/calibration.csv row 6, col `turn1_accuracy` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 popular mean_conf1_when_wrong | 0.772 | 0.771769 | describe_test/calibration.csv row 6, col `mean_conf1_when_wrong` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 popular share_wrong_with_conf1_over_0.9 | 0.234 | 0.234043 | describe_test/calibration.csv row 6, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 adversarial n_wrong | 46 | 46.000000 | describe_test/calibration.csv row 7, col `n_wrong` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 adversarial turn1_accuracy | 0.847 | 0.846667 | describe_test/calibration.csv row 7, col `turn1_accuracy` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 adversarial mean_conf1_when_wrong | 0.743 | 0.742745 | describe_test/calibration.csv row 7, col `mean_conf1_when_wrong` | OK |
| Results: H3 / Table 5 | Table 5 llava15_7b_fp16 adversarial share_wrong_with_conf1_over_0.9 | 0.239 | 0.239130 | describe_test/calibration.csv row 7, col `share_wrong_with_conf1_over_0.9` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding eir | 0.475 | 0.475032 | final_test/mitigations.csv row 2, col `eir` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding eir_minus_base | -0.050 | -0.049703 | final_test/mitigations.csv row 2, col `eir_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding eir_minus_base_ci_low | -0.057 | -0.057338 | final_test/mitigations.csv row 2, col `eir_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding eir_minus_base_ci_high | -0.043 | -0.043021 | final_test/mitigations.csv row 2, col `eir_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding ecr | 0.569 | 0.568627 | final_test/mitigations.csv row 2, col `ecr` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding ecr_minus_base | -0.013 | -0.012532 | final_test/mitigations.csv row 2, col `ecr_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding ecr_minus_base_ci_low | -0.035 | -0.035092 | final_test/mitigations.csv row 2, col `ecr_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding ecr_minus_base_ci_high | +0.011 | 0.011133 | final_test/mitigations.csv row 2, col `ecr_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding turn2_acc | 0.531 | 0.530741 | final_test/mitigations.csv row 2, col `turn2_acc` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding turn2_acc_minus_base | +0.042 | 0.041944 | final_test/mitigations.csv row 2, col `turn2_acc_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding turn2_acc_minus_base_ci_low | +0.035 | 0.034537 | final_test/mitigations.csv row 2, col `turn2_acc_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b grounding turn2_acc_minus_base_ci_high | +0.050 | 0.050093 | final_test/mitigations.csv row 2, col `turn2_acc_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook eir | 0.401 | 0.400955 | final_test/mitigations.csv row 3, col `eir` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook eir_minus_base | -0.124 | -0.123779 | final_test/mitigations.csv row 3, col `eir_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook eir_minus_base_ci_low | -0.135 | -0.134977 | final_test/mitigations.csv row 3, col `eir_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook eir_minus_base_ci_high | -0.113 | -0.113189 | final_test/mitigations.csv row 3, col `eir_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook ecr | 0.422 | 0.422464 | final_test/mitigations.csv row 3, col `ecr` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook ecr_minus_base | -0.159 | -0.158696 | final_test/mitigations.csv row 3, col `ecr_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook ecr_minus_base_ci_low | -0.186 | -0.186432 | final_test/mitigations.csv row 3, col `ecr_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook ecr_minus_base_ci_high | -0.129 | -0.128500 | final_test/mitigations.csv row 3, col `ecr_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook turn2_acc | 0.576 | 0.576481 | final_test/mitigations.csv row 3, col `turn2_acc` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook turn2_acc_minus_base | +0.088 | 0.087685 | final_test/mitigations.csv row 3, col `turn2_acc_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook turn2_acc_minus_base_ci_low | +0.076 | 0.076012 | final_test/mitigations.csv row 3, col `turn2_acc_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 qwen2vl_2b relook turn2_acc_minus_base_ci_high | +0.100 | 0.100278 | final_test/mitigations.csv row 3, col `turn2_acc_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding eir | 0.464 | 0.463642 | final_test/mitigations.csv row 4, col `eir` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding eir_minus_base | -0.013 | -0.012670 | final_test/mitigations.csv row 4, col `eir_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding eir_minus_base_ci_low | -0.017 | -0.016823 | final_test/mitigations.csv row 4, col `eir_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding eir_minus_base_ci_high | -0.009 | -0.008782 | final_test/mitigations.csv row 4, col `eir_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding ecr | 0.445 | 0.445122 | final_test/mitigations.csv row 4, col `ecr` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding ecr_minus_base | -0.014 | -0.013562 | final_test/mitigations.csv row 4, col `ecr_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding ecr_minus_base_ci_low | -0.031 | -0.030640 | final_test/mitigations.csv row 4, col `ecr_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding ecr_minus_base_ci_high | +0.006 | 0.005549 | final_test/mitigations.csv row 4, col `ecr_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding turn2_acc | 0.524 | 0.523889 | final_test/mitigations.csv row 4, col `turn2_acc` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding turn2_acc_minus_base | +0.009 | 0.008796 | final_test/mitigations.csv row 4, col `turn2_acc_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding turn2_acc_minus_base_ci_low | +0.005 | 0.005185 | final_test/mitigations.csv row 4, col `turn2_acc_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 grounding turn2_acc_minus_base_ci_high | +0.013 | 0.012778 | final_test/mitigations.csv row 4, col `turn2_acc_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook eir | 0.507 | 0.507149 | final_test/mitigations.csv row 5, col `eir` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook eir_minus_base | +0.031 | 0.030837 | final_test/mitigations.csv row 5, col `eir_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook eir_minus_base_ci_low | +0.022 | 0.022435 | final_test/mitigations.csv row 5, col `eir_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook eir_minus_base_ci_high | +0.038 | 0.037725 | final_test/mitigations.csv row 5, col `eir_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook ecr | 0.505 | 0.504902 | final_test/mitigations.csv row 5, col `ecr` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook ecr_minus_base | +0.046 | 0.046218 | final_test/mitigations.csv row 5, col `ecr_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook ecr_minus_base_ci_low | +0.020 | 0.019773 | final_test/mitigations.csv row 5, col `ecr_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook ecr_minus_base_ci_high | +0.069 | 0.068966 | final_test/mitigations.csv row 5, col `ecr_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook turn2_acc | 0.494 | 0.494444 | final_test/mitigations.csv row 5, col `turn2_acc` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook turn2_acc_minus_base | -0.021 | -0.020648 | final_test/mitigations.csv row 5, col `turn2_acc_minus_base` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook turn2_acc_minus_base_ci_low | -0.028 | -0.027687 | final_test/mitigations.csv row 5, col `turn2_acc_minus_base_ci_low` | OK |
| Results: mitigations / Table 6 | Table 6 llava15_7b_fp16 relook turn2_acc_minus_base_ci_high | -0.013 | -0.012683 | final_test/mitigations.csv row 5, col `turn2_acc_minus_base_ci_high` | OK |
| Results: mitigations / Table 6 | Prose net native qwen2vl_2b grounding | -0.337 | -0.337037 | final_test/mitigations.csv row 2, col `net_change_native_pi` | OK |
| Results: mitigations / Table 6 | Prose net native qwen2vl_2b relook | -0.296 | -0.295741 | final_test/mitigations.csv row 3, col `net_change_native_pi` | OK |
| Results: mitigations / Table 6 | Prose net native llava15_7b_fp16 grounding | -0.339 | -0.339444 | final_test/mitigations.csv row 4, col `net_change_native_pi` | OK |
| Results: mitigations / Table 6 | Prose net native llava15_7b_fp16 relook | -0.373 | -0.373333 | final_test/mitigations.csv row 5, col `net_change_native_pi` | OK |
| Results: mitigations / Table 6 | Prose net pi=0.5 qwen2vl_2b grounding | +0.047 | 0.046798 | final_test/mitigations.csv row 2, col `net_change_pi_0.5` | OK |
| Results: mitigations / Table 6 | Prose net pi=0.5 qwen2vl_2b relook | +0.011 | 0.010754 | final_test/mitigations.csv row 3, col `net_change_pi_0.5` | OK |
| Results: mitigations / Table 6 | Prose net pi=0.5 llava15_7b_fp16 grounding | -0.009 | -0.009260 | final_test/mitigations.csv row 4, col `net_change_pi_0.5` | OK |
| Results: mitigations / Table 6 | Prose net pi=0.5 llava15_7b_fp16 relook | -0.001 | -0.001123 | final_test/mitigations.csv row 5, col `net_change_pi_0.5` | OK |
| Results: mitigations / Table 6 | Dev no-suffix mean mass qwen2vl_2b | 0.299 | 0.299297 | phase6_dev.csv row 5, col `mean_mass_t2` | OK |
| Results: mitigations / Table 6 | Dev no-suffix min mass qwen2vl_2b | 0.000 | 0.000071 | phase6_dev.csv row 5, col `min_mass_t2` | OK |
| Results: mitigations / Table 6 | Dev no-suffix mean mass llava15_7b_fp16 | 0.582 | 0.582058 | phase6_dev.csv row 9, col `mean_mass_t2` | OK |
| Results: mitigations / Table 6 | Dev no-suffix min mass llava15_7b_fp16 | 0.033 | 0.033489 | phase6_dev.csv row 9, col `min_mass_t2` | OK |
| Results: failure cases | qwen2vl_2b all rows universal_failures | 517 | 517.000000 | failure_counts_test.csv row 2, col `universal_failures` | OK |
| Results: failure cases | qwen2vl_2b all rows images_with_failure | 105 | 105.000000 | failure_counts_test.csv row 2, col `images_with_failure` | OK |
| Results: failure cases | qwen2vl_2b a1 wrong n_rows | 1380 | 1380.000000 | failure_counts_test.csv row 4, col `n_rows` | OK |
| Results: failure cases | qwen2vl_2b a1 wrong universal_failures | 517 | 517.000000 | failure_counts_test.csv row 4, col `universal_failures` | OK |
| Results: failure cases | qwen2vl_2b a1 correct universal_failures | 0 | 0.000000 | failure_counts_test.csv row 3, col `universal_failures` | OK |
| Results: failure cases | llava15_7b_fp16 all rows universal_failures | 625 | 625.000000 | failure_counts_test.csv row 5, col `universal_failures` | OK |
| Results: failure cases | llava15_7b_fp16 all rows images_with_failure | 107 | 107.000000 | failure_counts_test.csv row 5, col `images_with_failure` | OK |
| Results: failure cases | llava15_7b_fp16 a1 wrong n_rows | 1428 | 1428.000000 | failure_counts_test.csv row 7, col `n_rows` | OK |
| Results: failure cases | llava15_7b_fp16 a1 wrong universal_failures | 624 | 624.000000 | failure_counts_test.csv row 7, col `universal_failures` | OK |
| Results: failure cases | llava15_7b_fp16 a1 correct universal_failures | 1 | 1.000000 | failure_counts_test.csv row 6, col `universal_failures` | OK |
| Discussion | 'about 87% of first answers are right' | 87% | 0.8722 / 0.8678 | describe_test/flip_summary.csv col `turn1_acc` | OK |
| Discussion | 'always agreeing costs 35 to 38 points' | 35–38 | Qwen 0.8722-0.4889=0.3833; LLaVA 0.8678-0.5150=0.3528 | describe_test/flip_summary.csv cols `turn1_acc`, `turn2_acc` | OK |
| Discussion | 'confidence and shift … AUROC 0.80 to 0.83' | 0.80–0.83 | conf 0.8011/0.8269, shift 0.8240/0.8259 | final_test/auroc.csv | OK |
| Discussion | 'a learned combination reaches 0.84 to 0.87' | 0.84–0.87 | 0.8333 to 0.8679 (LLaVA B 0.8333) | final_test/auroc.csv col `auroc_good_revision`, learned gates | MISMATCH: range is 0.83–0.87 (LLaVA learned B = 0.833) |
| Discussion | 'dev pattern … 23 to 34 wrong answers per split' | 23–34 | 23 to 34 | results/h3_calibration_dev.csv col `n_wrong` | OK |
| Discussion | 'one gate is reliably worse' (native) | one | see Sensitivity note: LLaVA conf (headline) and LLaVA shift (secondary) | final_test/gates.csv | NOTE: say 'one headline gate' |
| Limitations | 13%, 1%–4%, +0.009 | — | as above | as above | OK |
| Appendix A1/A2 | A1 flip neutral_doubt Qwen | 0.161 | 0.161111 | describe_test/flip_summary.csv row 5, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip neutral_doubt LLaVA | 0.374 | 0.374444 | describe_test/flip_summary.csv row 13, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip polite Qwen | 0.602 | 0.602222 | describe_test/flip_summary.csv row 6, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip polite LLaVA | 0.448 | 0.448148 | describe_test/flip_summary.csv row 14, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip authority Qwen | 0.529 | 0.528889 | describe_test/flip_summary.csv row 4, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip authority LLaVA | 0.487 | 0.487037 | describe_test/flip_summary.csv row 12, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip assertive Qwen | 0.836 | 0.835556 | describe_test/flip_summary.csv row 3, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip assertive LLaVA | 0.586 | 0.586296 | describe_test/flip_summary.csv row 11, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip random Qwen | 0.541 | 0.540833 | describe_test/flip_summary.csv row 9, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip random LLaVA | 0.458 | 0.458056 | describe_test/flip_summary.csv row 17, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip popular Qwen | 0.507 | 0.507500 | describe_test/flip_summary.csv row 8, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip popular LLaVA | 0.464 | 0.463611 | describe_test/flip_summary.csv row 16, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip adversarial Qwen | 0.547 | 0.547500 | describe_test/flip_summary.csv row 7, col `flip_rate` | OK |
| Appendix A1/A2 | A1 flip adversarial LLaVA | 0.500 | 0.500278 | describe_test/flip_summary.csv row 15, col `flip_rate` | OK |
| Appendix A1/A2 | A1 turn-1 acc random Qwen | 0.910 | 0.910000 | describe_test/flip_summary.csv row 9, col `turn1_acc` | OK |
| Appendix A1/A2 | A1 turn-1 acc random LLaVA | 0.913 | 0.913333 | describe_test/flip_summary.csv row 17, col `turn1_acc` | OK |
| Appendix A1/A2 | A1 turn-1 acc popular Qwen | 0.880 | 0.880000 | describe_test/flip_summary.csv row 8, col `turn1_acc` | OK |
| Appendix A1/A2 | A1 turn-1 acc popular LLaVA | 0.843 | 0.843333 | describe_test/flip_summary.csv row 16, col `turn1_acc` | OK |
| Appendix A1/A2 | A1 turn-1 acc adversarial Qwen | 0.827 | 0.826667 | describe_test/flip_summary.csv row 7, col `turn1_acc` | OK |
| Appendix A1/A2 | A1 turn-1 acc adversarial LLaVA | 0.847 | 0.846667 | describe_test/flip_summary.csv row 15, col `turn1_acc` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b conf1 auroc | 0.326 | 0.325713 | describe_test/flip_auroc_pooled_context.csv row 2, col `auroc` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b conf1 ci_low | 0.306 | 0.305665 | describe_test/flip_auroc_pooled_context.csv row 2, col `ci_low` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b conf1 ci_high | 0.347 | 0.346511 | describe_test/flip_auroc_pooled_context.csv row 2, col `ci_high` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 conf1 auroc | 0.473 | 0.473224 | describe_test/flip_auroc_pooled_context.csv row 14, col `auroc` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 conf1 ci_low | 0.452 | 0.451766 | describe_test/flip_auroc_pooled_context.csv row 14, col `ci_low` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 conf1 ci_high | 0.494 | 0.493917 | describe_test/flip_auroc_pooled_context.csv row 14, col `ci_high` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b reliance_blur auroc | 0.724 | 0.723907 | describe_test/flip_auroc_pooled_context.csv row 3, col `auroc` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b reliance_blur ci_low | 0.707 | 0.707434 | describe_test/flip_auroc_pooled_context.csv row 3, col `ci_low` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b reliance_blur ci_high | 0.739 | 0.739137 | describe_test/flip_auroc_pooled_context.csv row 3, col `ci_high` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 reliance_blur auroc | 0.710 | 0.710080 | describe_test/flip_auroc_pooled_context.csv row 15, col `auroc` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 reliance_blur ci_low | 0.698 | 0.697940 | describe_test/flip_auroc_pooled_context.csv row 15, col `ci_low` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 reliance_blur ci_high | 0.722 | 0.721955 | describe_test/flip_auroc_pooled_context.csv row 15, col `ci_high` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b reliance_blank auroc | 0.534 | 0.533867 | describe_test/flip_auroc_pooled_context.csv row 4, col `auroc` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b reliance_blank ci_low | 0.511 | 0.511219 | describe_test/flip_auroc_pooled_context.csv row 4, col `ci_low` | OK |
| Appendix A1/A2 | A2 qwen2vl_2b reliance_blank ci_high | 0.556 | 0.556492 | describe_test/flip_auroc_pooled_context.csv row 4, col `ci_high` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 reliance_blank auroc | 0.720 | 0.720131 | describe_test/flip_auroc_pooled_context.csv row 16, col `auroc` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 reliance_blank ci_low | 0.709 | 0.709133 | describe_test/flip_auroc_pooled_context.csv row 16, col `ci_low` | OK |
| Appendix A1/A2 | A2 llava15_7b_fp16 reliance_blank ci_high | 0.730 | 0.730297 | describe_test/flip_auroc_pooled_context.csv row 16, col `ci_high` | OK |
| Review (external) | MM-SY: ten tasks from TDIUC; three tones | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
| Review (external) | MM-SY: up to five rounds raise rate by < 5 points | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
| Review (external) | MM-SY: higher in the larger model of two pairs | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
| Review (external) | MM-SY: LLaVA-1.5 94.6% | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
| Review (external) | PENDULUM: ~2,000 pairs, six domains | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
| Review (external) | Rabby et al.: ten VLMs; EIR/ECR | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
| Review (external) | Perez et al. 2023; Sharma et al. 2024; Gupta et al. (workshop) | — | not in results/ | source papers | UNTRACEABLE to results/ (external claim; must be checked against the paper) |
