# AI use log

Record of what Claude Code was used for in each session and what the team verified. The final report's AI Use Statement is based on this file.

## 2026-11-07: Phase 0 setup

**Claude Code was used for:**
- Reading `CLAUDE.md`, `SPEC.md` and the TC1 user guide
- Creating the repo layout, `.gitignore`, `README.md` skeleton, `environment.yml` notes and `refs.bib` placeholder
- Writing `slurm/job_smoke.sh`
- Testing the logic of the smoke script on a laptop with stand-ins for the cluster commands. There was no GPU and no real SLURM, so this does not replace the smoke job.

**Team verification:**
- [ ] Smoke job run on TC1 and its output reviewed
- [ ] Team has read and can explain `slurm/job_smoke.sh`

## 2026-11-10: Move to Kaggle; Phases 1–2; Kaggle runner

**Claude Code was used for:**
- Rewriting the compute section of `CLAUDE.md` for Kaggle and recording the scope cut in `CLAUDE.md` and `SPEC.md`
- Inspecting the POPE files (format, labels, images shared across splits) before writing the loader
- Checking Hugging Face licence, gating and revisions for the candidate models
- Writing `data.py`, `splits.py`, `prompts.py`, `models.py` (with `MockVLM`), `pipeline.py`, `metrics.py`, `paths.py`, the scripts and the configs
- Writing 43 CPU tests and dry-running every script with the mock model on the real dev data
- Generating `scripts/kaggle_runner.ipynb`. Only its dev-loop cell was run locally, with the mock model. The rest is untested until it runs on Kaggle.

**Team verification:**
- [ ] Read the subsample design (SPEC.md "Changes") and `results/splits_summary.csv`
- [ ] Read the pushback templates in `configs/pushback.yaml`
- [ ] Hand-check the 5 printed pilot examples per model (exact prompt, p(yes), p(no), a1, a2)
- [ ] Can explain `models.py` (yes/no from logits) and `pipeline.py` (resume, shards)

## 2026-11-10 (evening): TC1 reachable; pilot; dev baseline and signals

**Claude Code was used for:**
- Connecting to TC1 from the laptop on campus. The user created the SSH key and typed their password into `ssh-copy-id` themselves; Claude Code never saw or stored it. Claude Code then used key-based SSH to copy code/data, install packages on the head node (install only, no code run there) and submit SLURM jobs.
- Debugging the TC1 environment: conda failed on the NFS home (symlink loop), then uv's cache failed (no file locks on NFS); the fix is a uv venv with the cache in `/tmp`. Writing `requirements-gpu.txt` (exact versions read from the TC1 venv), the uv setup in `README.md` and `environment.yml`.
- Writing `slurm/job_pilot.sh` and `slurm/job_dev.sh`, running the smoke test (job 73332), the pilot (job 73333) and the LLaVA fp16 pilot (job 73335), and summarising `results/pilot.csv`.
- After the user's decisions (suffix "Answer yes or no." in the main runs; both models; LLaVA in fp16 if its pilot flip rate stayed within 20-70%): updating `configs/pushback.yaml`, SPEC.md section 4, adding `configs/models/llava15_7b_fp16.yaml`, and a guard so a resumed file cannot mix two suffix settings.
- Writing `signals.py` (conf1, blur and blank image reliance, post-pushback shift), `stats.py` (AUROC, bootstrap clustered by image) and `scripts/run_signals.py`, with 8 new CPU tests (52 in total), and dry-running baseline -> merge -> signals end to end with the mock model.
- The blur strength (sigma = 3% of the longer image side) was chosen by Claude Code before any signal was computed and is not tuned; the team should confirm it is reasonable.

**Team verification:**
- [ ] Review the pilot table (`results/pilot.csv`) and the printed examples in `results/logs_tc1/`
- [ ] Check that the Qwen pilot on TC1 matches the earlier Kaggle pilot (turn-1 acc 0.87, flip 0.47, mass 0.38 on both)
- [ ] Read `signals.py` and the signal definitions; check the AUROC hand-computed tests in `tests/test_signals.py`
- [ ] Look at a few blurred and blank images to confirm the degradation is sensible
- [ ] Confirm `results/signals_auroc_dev.csv` was produced by `scripts/run_signals.py` on dev only

## 2026-11-10 (night): Phase 5 on dev (CPU), Phase 6 queued (TC1)

**Claude Code was used for:**
- Writing `results/analysis_plan.md` to the user's rules (H2 and H3 decision rules, sign conventions, grouped CV, π sensitivity). It was committed (c18dfe8) before any gate was fitted. The plan discloses that the dev signal AUROCs had already been seen and that this informed the orientation of the shift gate, which departs from SPEC.
- Writing `src/vlmsyc/gates.py` (EIR/ECR, threshold sweep, oracle and random gates, image-grouped folds, learned logistic gate), `scripts/fit_gates_dev.py`, `scripts/make_figures.py` and 6 hand-computed tests in `tests/test_gates.py` (60 tests in total). It also removed the circular flip AUROC of `shift_to_claim` from `run_signals.py`.
- Running the analysis on the laptop CPU, on dev only. A quick 30-resample run in a scratch folder was done first to check that the code worked. Nothing in the analysis was changed after it, and the official 1,000-resample run (code at 057c2ef, re-run at b2e089e to record a clean commit) gave identical CSVs.
- Finding and reporting that an earlier claim ("image-driven answers are easier to push over") was a confound with the turn-1 answer (Simpson's paradox). The claim is withdrawn in the overview.
- Phase 6 code: the grounding / re-look / no-suffix conditions in `configs/pushback.yaml` and `prompts.py`, several images per conversation in `models.py`, a check that the chat template keeps the condition's text, and `slurm/job_phase6.sh` (TC1 job 73360).
- Updating `report/PROJECT_OVERVIEW.md` with the Phase 5 results, the correction, and the pending items.

**Team verification:**
- [ ] Read `results/analysis_plan.md` and agree with the decision rules before the test run
- [ ] Re-derive one EIR/ECR value by hand from `results/signals_dev_qwen2vl_2b.jsonl`
- [ ] Check the oracle and random-gate logic in `gates.py` and `tests/test_gates.py`
- [ ] Look at `figures/eir_ecr_dev.png`, `figures/sensitivity_pi_dev.png` and `figures/reliability_by_split_dev.png`
- [ ] Confirm H2 = mixed and H3 = not supported are reported as such (including the negative result)
- [ ] Check the printed rendered prompts in the Phase 6 job log (system text and second image present)

## 2026-11-11 (01:00–01:30): Phase 6 on dev scored

**Claude Code was used for:**
- Monitoring TC1 job 73360 (all six model × condition runs complete, all merge checks clean) and copying results and logs back.
- Writing and running `scripts/score_phase6_dev.py`, which scores each condition on the EIR/ECR/net-change axes with paired, image-clustered bootstrap differences to base (`results/phase6_dev.csv`).
- Adding the mitigation points to `figures/eir_ecr_dev.png` and writing the Phase 6 section of `report/PROJECT_OVERVIEW.md`. This includes the note that no-suffix turn-2 readings have low yes/no mass and are not trustworthy, and that Qwen's template adds a default system prompt.

**Team verification:**
- [ ] Read the rendered prompts for each condition in `results/logs_tc1/output_phase6_73360.out`
- [ ] Check one row of `results/phase6_dev.csv` against the JSONL by hand

## 2026-11-11: Phase 7 preparation (no test data touched)

**Claude Code was used for:**
- Writing the team's headline choices and the confirmatory H2 rule into `results/analysis_plan.md` as Amendment A (commit c043521), before any final-run code. It records that learned gate A gained pushback type after the first dev analysis.
- Adding pushback type to the learned gates, re-running the dev cross-validation and refreezing `results/frozen_gates.json` (commit 90c5f8d). The earlier dev H2 table is kept as `results/h2_dev_before_amendment_A.csv`.
- Writing `src/vlmsyc/release.py` (tag and clean-tree check, never-overwrite and header-matched resume), `scripts/run_final.py`, `scripts/analyse_final.py` (frozen gates only; references; matched random gate; never-accept comparison; native π plus 0.1, 0.3 and 0.5) and `slurm/job_final.sh`, with 8 new tests (69 in total).
- Running the dev dry run (`results/final_dryrun_dev/`, `figures/final_dryrun_dev/`, labelled in-sample and not a result).
- Updating `report/PROJECT_OVERVIEW.md` and `README.md`. The `pre-final` tag was **not** created: it waits for team review.

**Team verification (before the tag):**
- [ ] Read Amendment A in `results/analysis_plan.md` and agree with it
- [ ] Read `scripts/run_final.py` and `src/vlmsyc/release.py`; try `python scripts/run_final.py --final` on an untagged HEAD and see it refuse
- [ ] Inspect `results/final_dryrun_dev/*.csv` and `figures/final_dryrun_dev/*.png`
- [ ] Agree that `results/frozen_gates.json` (sha256 71979e78…) is what will be applied to test

## 2026-11-11 (02:30–03:00): GPU rehearsal, two-job split, GitHub backup

**Claude Code was used for:**
- Pre-push checks (no PDF or data ever committed, no secrets in the tree or history, sizes), then adding the private GitHub remote. GitHub's auto-README commit was merged in (not force-pushed over). Pushed with the user's existing git credentials; no credentials were typed or stored.
- Syncing TC1 results and logs (`.err` logs committed; the duplicate single-shard JSONL files left out), and checking that TC1's code matches commit a3424af.
- Splitting the final run into job A (base + signals) and job B (grounding/re-look + analysis) with `run_final.py --steps` and `slurm/submit_final.sh`, plus a `MODE=dev` GPU rehearsal. The rehearsal ran from a fresh clone on TC1 (jobs 73378 and 73379) and reproduced the earlier dev answers exactly.
- Reading MM-SY (arXiv 2410.11302) from the arXiv HTML through a web tool that summarises pages, and adding its correction metric (Cor) and stubbornness finding to the overview. The quotes come from that summary tool and are flagged for checking against the PDF. Nothing was added to refs.bib.

**Team verification:**
- [ ] Check the MM-SY quotes and Table 2 numbers in the overview against the PDF, then add the refs.bib entry
- [ ] Read `results/logs_tc1/` rehearsal output (or TC1 `~/final_rehearsal/output_finald*`) before approving the tag

## 2026-11-11: Amendment B

**Claude Code was used for:**
- Writing Amendment B, the team's rules for combining the two models for H2 (including "reversal") and H3, into `results/analysis_plan.md` (commit 0e7d3ff) before any code change.
- Adding `ci_side`, `overall_h2` and `overall_h3` to `gates.py`, with a test covering every case, and using them in `analyse_final.py` (labels printed and saved in `run_info.json`) and `fit_gates_dev.py`. `fit_gates_dev.py` was deliberately not re-run, so `frozen_gates.json` keeps sha256 71979e78… (checked before and after).
- Re-running the dev dry run: H2 overall "mixed", H3 overall "not supported" (in-sample, not a result).

**Team verification:**
- [ ] Read Amendment B and `tests/test_gates.py::test_amendment_b_combining_rules`

## 2026-11-11: Team review before the `pre-final` tag

**Review record, as reported by the user (Manasi):** Manasi reviewed all the files and scripts. Claude Code recorded this statement as given; it did not take part in the review.

**Claude Code was used for:** adding Note C to `results/analysis_plan.md` (the full sha256 of `frozen_gates.json` applied to test, and the team's rule for analysis-script failures), and re-running the pre-tag checks.

## 2026-11-11 13:09: `pre-final` tag and final-run submission

**Claude Code was used for:**
- Re-checking the four preconditions from scratch, all passing:
  1. Amendment B committed;
  2. `frozen_gates.json` sha256 71979e78… identical to the plan's Note C;
  3. clean tree, 71/71 tests pass, review record present;
  4. only the already-reported commit 94e8994 touched src/scripts/slurm since the rehearsal.
- Creating the annotated tag `pre-final` on commit 8269e4ff3f08a459246c94b27230733a5371189d and pushing `main` and the tag to the private GitHub remote with the user's existing credentials.
- Bundling `main` and `pre-final`, cloning into a fresh `~/final` on TC1, and checking HEAD = tag, the clean tree, and that the `splits.json` and `frozen_gates.json` hashes match.
- Submitting `bash slurm/submit_final.sh` at 2026-11-11 13:09:37 +08:00: job A 73424 (base + signals), job B 73425 (grounding, re-look, analysis; afterok on A).
- No test outputs have been read. No code or config was changed after the tag; only this log and the overview were edited.

## 2026-11-11 (during the test run): post-final preparation on branch `post-final-prep` (dev data only)

**Claude Code was used for:**
- `scripts/make_failure_gallery.py`, with its selection rule fixed in the docstring before any test output is read:
  - a row is a universal failure if all 6 frozen gates (native-π threshold) and both prompt mitigations give a wrong final answer;
  - the top 6 images per model are shown, ranked by count.

  Run on dev: `results/failure_gallery_dev.csv`, `results/failure_counts_dev.csv`, `figures/failure_gallery_dev_*.png`.
- `scripts/make_zip.sh`, which builds from committed files with `git archive`. It excludes `data/`, PDFs, weights, caches and logs, includes `results/` and `figures/`, and checks the exclusions. Test build: 121 files, 84.1 MB uncompressed.
- The README reproduction section, with runtimes taken from SLURM `sacct`.
- `report/test_number_checklist.md`: where every test number for the report comes from, plus three gaps the tagged analysis does not produce.
- No test outputs were read, no tagged code or config was changed (new files only), and only job state was checked on TC1.

**Team verification:**
- [ ] Agree with the gallery selection rule before it is run on test
- [ ] Decide whether to add the three descriptive test tables listed as gaps in `report/test_number_checklist.md`

## 2026-11-11 (during the test run): Amendment C and `describe_final.py` (branch `post-final-prep`, no test output read)

**Claude Code was used for:**
- Checking `data/splits.json`: it holds only the seed, the dev fraction, `n_images`, and image ids and question ids per partition, with no labels. Including it in the ZIP, and adding a guard so `make_zip.sh` fails if `src/`, `configs/` or `slurm/` differ from `pre-final` (tested: it refuses an older commit).
- Writing Amendment C (commit c33404d), as specified by the team, including the disclosure that it was written after the tag but before any test output was read.
- Writing `scripts/describe_final.py` (commit d898408). It reuses the dev functions unchanged and mirrors the dev loops. It refuses to run on test until `results/final_test/run_info.json` exists, and never overwrites. On dev it reproduces the existing tables exactly (by-answer 12/12, pooled 24/24, calibration 6/6, reliability 60/60 rows).
- **Error found and reported:** commit 92c8c6a reintroduced 16 circular `shift_to_claim`/`abs_shift` flip rows into `results/signals_auroc_dev.csv`, because a later rsync from TC1 copied back an older file. They were never used in the overview. The fix is pending a team decision.

## 2026-11-11: Fix of the stale `results/signals_auroc_dev.csv` (branch `post-final-prep`, dev only)

**Mistake (Claude Code's):** in commit 92c8c6a (Phase 6 results), an `rsync` of `results/` from TC1 replaced the local `signals_auroc_dev.csv` with TC1's older copy. That copy still had the 16 circular rows (`shift_to_claim`/`abs_shift` predicting flips), which had been removed in 057c2ef. It was committed without noticing, and found by the dev-equality check of `describe_final.py`.

**Fix (team-chosen option (a), adjusted):**
1. Ran `run_signals.py --join-only` for both models (dev).
2. Restored `results/signals_dev_*.jsonl` from git, so their headers keep the original commit, seed and versions; `git diff` on them is empty.
   - The regenerated rows differed from the committed ones in 2 of 7,200 rows per model, in `shift`/`shift_to_claim` only, by at most 4.4e-16 (Linux on TC1 vs macOS floating-point rounding).
   - The CSV was therefore rebuilt from the **restored committed rows** with the unchanged `run_signals.auroc_table` and `write_auroc_csv`. The result is byte-for-byte the same as the `--join-only` output.
3. **Checks:** the new CSV has 0 circular flip rows (16 before). All 64 remaining rows are identical, in every column, to the corresponding old rows, including the 24 conf1/blur/blank flip rows. `frozen_gates.json` sha256 is still 71979e78b8cd69a6281e3c04ef0c3502b1b62ff20473196ffd2caab39ccebe86.

**Audit of every file in 92c8c6a:**
- The six `{grounding,relook,nosuffix}_dev_*.jsonl` files, both `signals_dev_*.jsonl` files and `logs_tc1/output_phase6_73360.out` are sha256-identical to TC1's `~/project` copies.
- `phase6_dev.csv`, at 92c8c6a and at HEAD, is identical to a fresh `score_phase6_dev.py` run.
- The three dev figures at HEAD are identical to a fresh `make_figures.py` run.
- `AI_USE_LOG.md`, `PROJECT_OVERVIEW.md` and `make_figures.py` were hand edits in that commit.
- **The `signals_dev_*.jsonl` files at 92c8c6a are TC1's original GPU-job outputs** (header commit fb22212). They replaced a locally re-joined copy whose rows differed only in that floating-point rounding. The frozen gates were fit later, on these committed files. Not stale; recorded for completeness.
- **No other stale file was found.**
