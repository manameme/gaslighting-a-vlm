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
