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
