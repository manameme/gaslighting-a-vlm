# CLAUDE.md — SC4001 group project: when should a VLM accept a user's correction?

Read this file fully at the start of every session. The detailed build spec is in `SPEC.md`.

## 1. Project in one paragraph
Course: NTU SC4001 Neural Networks and Deep Learning, group project (Direction E: hallucination and reliability in VLMs).
Deadline: **16 Nov 2026, 11:59 PM** (SGT). Late penalty 5% per day. Report: PDF, A4, single column, Arial 10pt, max 10 pages excluding references, cover page, contents page and appendix. Code goes in a ZIP with a README that has reproduction instructions.

Study: a small open VLM answers a POPE yes/no question (turn 1), the user pushes back (turn 2), and we study when the model should change its answer.
Core question: **which signal should decide whether a VLM accepts a correction?** We compare "gates" (rules that allow or block a revision) using turn-1 confidence, image reliance (confidence drop when the image is blurred or blanked), crop consistency, and post-pushback logit shift. Every gate is compared against a random gate (same revision rate) and an oracle gate.

**Scope cut (2026-11-10, Kaggle compute):** about 500 questions per POPE split, 1 to 2 models, fp16 or 4-bit. Crop consistency and the contrastive-decoding baselines are dropped. If more must go, ask the user first.

Hypotheses:
- H1: turn-1 confidence predicts flips.
- H2: image-reliance signals add information beyond confidence.
- H3: on POPE's adversarial split models are confidently wrong, so confidence gates degrade while image-reliance gates hold up.

Novelty claim is modest and comes from a limited literature check. Do not claim we are first to show VLMs cave under pushback. Related work to read and cite correctly: MM-SY (arXiv 2410.11302), Zhao et al. (arXiv 2408.11261), PENDULUM (arXiv 2512.19350), Moral Sycophancy in VLMs (arXiv 2602.08311, source of EIR/ECR), and a NeurIPS 2026 workshop paper "Are You Sure You're Sure?" (a text-only template-confound audit of one benchmark).

## 2. Non-negotiable rules

### Honesty and the grading rubric
1. **Never fabricate results, numbers, citations or paper details.** Every number in the report must come from a file in `results/` produced by a script in this repo. If a number is not in `results/`, say so.
2. **Citations:** only add entries to `refs.bib` that are in the verified list above, or that the user confirms after reading the source. Mark anything else `% UNVERIFIED` and tell the user. The course deducts marks for hallucinated references.
3. **Test set discipline:** the test split is touched exactly once, by `run_final.py` with the `--final` flag. All tuning (thresholds, gate choice, prompts) uses the dev split only. Code must enforce this (see SPEC.md section 2); do not bypass it.
4. **Report only experiments that were actually run.** Failed or negative results are valid and must be reported.
5. **AI Use Statement:** keep `AI_USE_LOG.md` updated with what Claude Code was used for in each session (coding, debugging, plotting, drafting) and what the team verified. The final report must include an honest AI Use Statement.
6. The team must be able to explain every line. Prefer simple, readable code with comments over clever code.

### Compute: free Kaggle GPUs (changed 2026-11-10; TC1 could not be reached)
- GPU: one NVIDIA T4, 16 GB, in a Kaggle notebook. T4 has no fast bf16, so use fp16, or 4-bit (bitsandbytes) for models that do not fit in fp16. Do not assume FlashAttention-2 works. Use PyTorch's default attention.
- Runner: `scripts/kaggle_runner.ipynb`. It installs pinned packages, prints versions, runs the pilot, then runs the dev split in a resumable loop. Kaggle needs a phone-verified account for internet and GPU.
- Sessions end, and the GPU quota is weekly, so check the quota shown in Kaggle. Every run must be **resumable**: run scripts take `--resume` and skip rows already in the output JSONL. Write output after every batch.
- Storage: everything written goes under `RESULTS_DIR`, an environment variable with default `results/`. Kaggle cannot mount Google Drive, so the runner can copy results to and from Drive with `rclone` (set-up in README). On Colab, `RESULTS_DIR` can point at the mounted Drive directly.
- Parallel work: run scripts take `--shard k/n`, so teammates can each run one shard on their own Kaggle account. Each shard writes its own file. `scripts/merge_shards.py` combines them and reports duplicates and missing rows.
- Kaggle downloads data at run time: POPE json from GitHub at a pinned commit, and only the COCO val2014 images POPE references (`scripts/download_data.py`).
- Results produced on Kaggle are copied back into `results/` in the repo and committed.

### How Claude Code works with Kaggle
- Claude Code runs on the user's laptop. It writes code, the runner notebook, tests and docs. Locally it runs only CPU-only unit tests in `.venv` with the mock model, never real weights.
- Claude Code never asks for, types, stores or logs passwords, Kaggle tokens or rclone tokens. Secrets go in Kaggle's Secrets panel and are typed by the user.
- Before telling the user to start a long run, state: what it runs, expected time (from the pilot), output file, and how to resume.
- After a run, read the notebook output and the result files first; diagnose before re-running.

## 3. Repo layout (create exactly this)
```
.
├── CLAUDE.md  SPEC.md  README.md  AI_USE_LOG.md  refs.bib
├── configs/            # one yaml per model + experiment grid
├── data/               # NOT committed: POPE json, COCO images (see README)
├── src/vlmsyc/
│   ├── data.py  splits.py  models.py  prompts.py  pipeline.py
│   ├── signals.py  gates.py  metrics.py  stats.py
├── scripts/
│   ├── download_data.py  build_splits.py  pilot.py  run_baseline.py  run_signals.py
│   ├── merge_shards.py  fit_gates_dev.py  run_final.py  make_figures.py
│   ├── kaggle_runner.ipynb
├── slurm/              # unused since the move to Kaggle (kept for history)
├── tests/              # CPU-only, mock model
├── results/            # JSONL + CSV, committed (small)
├── figures/
└── report/             # LaTeX or docx source, Arial 10pt, A4
```

## 4. Workflow (follow in order; stop and report at each checkpoint)
Work in small commits with clear messages. Before each phase, restate the goal and the acceptance check. After each phase, update `AI_USE_LOG.md`.

**Phase 0 — Setup (30 min).** Create the layout, `environment.yml` notes and README skeleton. The environment check (GPU visible, `torch.cuda.is_available()`, `transformers` imports, versions) is now the first cell of `scripts/kaggle_runner.ipynb`. `slurm/job_smoke.sh` was written for TC1 and never run.

**Phase 1 — Data and splits.** `data.py`, `splits.py`, `scripts/build_splits.py`. Image-level dev/test split. Unit tests.
Checkpoint: `results/splits_summary.csv` shows counts per split and POPE category, no image in both dev and test.

**Phase 2 — Model wrapper and two-turn pipeline.** `models.py`, `prompts.py`, `pipeline.py`, plus a mock model for tests.
Checkpoint: 5 hand-checked examples per candidate model: print the exact chat-template text, p(yes), p(no), turn-1 answer, turn-2 answer.

**Phase 3 — Pilot.** `scripts/pilot.py` on 100 dev questions per model. Report flip rate and seconds per example. Decide with the user which 2 to 3 models to use (target flip rate roughly 20–70%).
Checkpoint: pilot table in `results/pilot.csv`.

**Phase 4 — Baseline runs (dev only).** All pushback types × paraphrases × splits, turn-1 correct and incorrect. Shards are merged with `scripts/merge_shards.py`.
Checkpoint: `results/baseline_dev_<model>.jsonl` complete; flip-rate table.

**Phase 5 — Signals and gates (dev only).** Compute signals (conf1, blur and blank image reliance, shift; crop consistency is cut), fit learned gate on dev, tune thresholds on dev, build EIR–ECR curves with random and oracle bounds.
Checkpoint: dev figures and the AUROC table.

**Phase 6 — Mitigation baselines and robustness.** Grounding prompt, re-look prompt, instruction-placement robustness check. Contrastive decoding (blurred-image and leading-query) is cut.

**Phase 7 — Final test run (once).** Freeze configs in a git tag `pre-final`. Run `run_final.py --final`. Never rerun to improve numbers; if a bug is found, report it and discuss with the user first.

**Phase 8 — Figures, tables, failure-case gallery.** `make_figures.py` writes everything the report needs. Failure gallery: images where every method fails, with question, claimed answer, model outputs.

**Phase 9 — Reproducibility and report support.** Clean README (dependencies, data prep, exact commands, expected runtimes), test from a fresh clone, produce the ZIP. Help draft the report only from files in `results/`; mark every unverified claim.

## 5. Coding standards
- Python 3.10+, type hints on public functions, docstrings explaining the why. Comment code for graders.
- Fixed seeds (default 0), stored in every output row. Log package versions in each result file header.
- JSONL outputs: one row per (question_id, model, pushback_type, paraphrase_id, condition). Flush after each batch. Files live under `RESULTS_DIR`.
- Batch inference with fp16 (or 4-bit weights with fp16 compute). Catch CUDA out-of-memory and reduce batch size instead of crashing.
- No hard-coded absolute paths. Use `configs/*.yaml` and environment variables.
- Tests in `tests/` run on CPU in under 1 minute with the mock model: splits, prompt building, yes/no probability extraction, metrics (EIR, ECR, AUROC, random and oracle gates), resume logic.

## 6. When unsure
Ask the user one short question rather than guessing, especially about: model choice, anything that would touch the test split, anything that costs more than about 1 GPU-hour, and any claim about prior work.
