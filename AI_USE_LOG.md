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
