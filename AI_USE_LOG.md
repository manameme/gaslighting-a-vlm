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
