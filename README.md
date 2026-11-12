# When should a VLM accept a user's correction?

NTU SC4001 Neural Networks and Deep Learning, group project (Direction E: hallucination and reliability in VLMs).

A small open VLM answers a POPE yes/no question, the user pushes back, and we compare rules ("gates") that decide whether the model may change its answer. See `CLAUDE.md` for the study design and `SPEC.md` for the build spec (its "Changes" section lists the current scope).

**Status:** Phases 1–3 done (data, splits, model wrapper, two-turn pipeline, pilot). Phases 4–5 (dev baseline and signals) run on the NTU CCDS TC1 cluster (Tesla V100 32 GB). Kaggle T4 GPUs remain a fallback.

## Repo layout

| Path | Contents |
| --- | --- |
| `configs/` | `data.yaml`, `experiment.yaml`, `pushback.yaml`, one yaml per model in `models/` |
| `data/` | Downloaded POPE json and COCO images (not committed); `splits.json` (committed) |
| `src/vlmsyc/` | Library code: `data`, `splits`, `prompts`, `models`, `pipeline`, `metrics`, `signals`, `stats`, `paths` |
| `scripts/` | `download_data.py`, `build_splits.py`, `pilot.py`, `run_baseline.py`, `merge_shards.py`, `run_signals.py`, `kaggle_runner.ipynb` |
| `tests/` | CPU-only tests using a mock model |
| `results/` | JSONL and CSV outputs, committed |
| `slurm/` | SLURM job scripts for TC1: `job_smoke.sh`, `job_pilot.sh`, `job_dev.sh`, `job_phase6.sh`, `job_final.sh` |
| `requirements-gpu.txt` | Exact package versions of the GPU environment |

## Where outputs go

All outputs go to `RESULTS_DIR` (environment variable, default `results/`). Inputs come from `DATA_DIR` (default `data/`).
Run scripts append rows and force them to disk after every batch, take `--resume` to skip finished rows, and take `--shard k/n` (k from 0) so that n people can each run a disjoint part. Each shard writes `baseline_dev_<model>_shard<k>of<n>.jsonl`. `scripts/merge_shards.py --model <model>` merges them into `baseline_dev_<model>.jsonl` and reports missing shards, duplicates and conflicts.

## Laptop setup (CPU only)

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements-cpu.txt   # numpy, pillow, pyyaml, scikit-learn, matplotlib, pytest (pinned)
./.venv/bin/python -m pytest -q                  # 71 tests, a few seconds (3 skip if POPE/COCO data are not downloaded)
```
This environment runs everything except model inference: the tests, the gate fitting on dev, the final and descriptive analyses of saved outputs, the figures, and the ZIP.

## Data preparation

```bash
python scripts/download_data.py   # POPE json at a pinned commit + the 500 COCO val2014 images it uses (~80 MB)
python scripts/build_splits.py    # only needed to rebuild data/splits.json (already committed)
```

- POPE: github.com/AoiDragon/POPE, commit `109985c74a00a88ba5341f9b0c1d4900f2a1f789`, MIT licence. We use the three COCO files (`random`, `popular`, `adversarial`).
- COCO val2014 images from images.cocodataset.org. COCO annotations are CC BY 4.0; the images come from Flickr under their own licence terms (see cocodataset.org/#termsofuse). Only the 500 images POPE references are downloaded.
- Split: 500 images → 200 dev / 300 test by image. Each image gives one question per POPE split, with yes/no balanced, so there are 500 questions per POPE split (dev 200 per split). See `results/splits_summary.csv`.

## Running on TC1 (NTU CCDS GPU cluster) — main setup

TC1 is reachable only from the NTU network (on campus or via the NTU VPN). Read the cluster's user guide first; in particular, **no code may run on the head node**: everything runs through `sbatch`. Our QoS allows 1 GPU, 2 submitted jobs and 6 h per job, so jobs run one at a time and every step is resumable.

1. **SSH key (once).** Log in with your NTU account; do not store the password anywhere.
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_tc1
   ssh-copy-id -i ~/.ssh/id_ed25519_tc1.pub <ntu-id>@10.96.189.11
   ```
   Then add a `Host tc1` entry with that `IdentityFile` to `~/.ssh/config`.
2. **Environment (once, on the head node: installing only).** Conda does not work on the NFS home directory (symlink loop), and uv's cache needs file locks that NFS lacks, so use a uv venv with the cache in `/tmp`:
   ```bash
   module load python/3.11 uv
   export UV_CACHE_DIR=/tmp/$USER-uvcache UV_LINK_MODE=copy
   uv venv --python "$(which python3.11)" ~/venvs/vlmsyc
   source ~/venvs/vlmsyc/bin/activate
   uv pip install -r requirements-gpu.txt --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/cu124
   ```
3. **Code and data.** Copy the repo (with `data/`, without `.git`) and record the commit:
   ```bash
   rsync -az --exclude .venv --exclude .git --exclude '*.pdf' ./ tc1:~/project/
   git rev-parse HEAD | ssh tc1 'cat > ~/project/.git_commit'
   ```
   Model weights download inside the job into `~/hf_cache` (compute nodes have internet).
4. **Jobs** (from `~/project` on TC1):
   ```bash
   sbatch slurm/job_smoke.sh     # GPU, CUDA, fp16 and internet check (~1 min)
   sbatch slurm/job_pilot.sh     # Phase 3 pilot, 100 dev questions per model and prompt variant
   sbatch slurm/job_dev.sh       # Phases 4-5: dev baseline, merge, signals, for both models
   squeue -u $USER               # status; logs are output_<job>_<id>.out / error_<job>_<id>.err
   ```
   If a job times out, submit the same command again: every step uses `--resume`.
5. Copy results back and commit them: `rsync -az tc1:~/project/results/ results/`.

## Phase 7: the one test run (TC1)

Only after the team has reviewed `results/analysis_plan.md` (including Amendment A) and the dry runs.
1. On the laptop: `git tag pre-final && git push origin main pre-final && git bundle create sc4001.bundle main pre-final && scp sc4001.bundle tc1:~/`
2. On TC1, a fresh clone of the tagged commit (TC1 has no GitHub credentials, so we use the bundle):
   ```bash
   git clone -b main ~/sc4001.bundle ~/final && cd ~/final && git checkout pre-final
   bash slurm/submit_final.sh        # job A: base + signals; job B (waits for A): grounding, re-look, analysis
   ```
   **Where the data comes from:** a fresh clone contains only `data/splits.json`. The POPE files and the 500 COCO images stay in `~/project/data` (downloaded earlier by `scripts/download_data.py`). The job sets `DATA_DIR=$HOME/project/data`, and `run_final.py --final` refuses to start unless that folder's `splits.json` is byte-identical to the committed one.
3. After a timeout, resubmit only the unfinished part, e.g. `sbatch --export=ALL,PART=A,RESUME=1 slurm/job_final.sh`. It resumes only if every header matches.
4. Copy back `results/*_test_*.jsonl`, `results/final_test/` and `figures/final_test/`, and commit them. Never re-run to change numbers.

**Rehearsals (no tag needed, never read the test split):**
- On a laptop, CPU only: `python scripts/run_final.py --dry-run-dev --resume` reuses the dev files.
- On TC1, GPU, with the same two jobs and the real models on 30 dev questions: `MODE=dev OUT_DIR=$HOME/devtest bash slurm/submit_final.sh`.

## Running on Kaggle (fallback)

1. **Put the code where Kaggle can get it.** Either push this repo to a **private** GitHub repo and add a Kaggle secret `GITHUB_TOKEN` (a fine-grained token with read-only access to that repo), or upload the repo folder as a private Kaggle dataset and set `CODE_DATASET` in the notebook.
2. **Optional: copy results to Google Drive.** Kaggle cannot mount Drive, so the notebook uses `rclone`. Set it up once on a laptop:
   ```bash
   brew install rclone
   rclone config      # new remote named "gdrive", storage "drive", scope "drive.file", log in when the browser opens
   rclone mkdir gdrive:sc4001-vlm/results
   rclone config show gdrive   # copy everything it prints into a Kaggle secret named RCLONE_CONF
   ```
   The rclone config holds a login token for that Google account. Keep it only in Kaggle Secrets. Without it, results stay in `/kaggle/working/results`; download them before the session ends or use *Save Version*.
3. **In Kaggle:** create a notebook from `scripts/kaggle_runner.ipynb` (*File → Import Notebook*). Set *Accelerator* to GPU T4 and *Internet* to on, edit the settings cell, then *Run All*. It installs `transformers==4.57.6`, `accelerate==1.10.1` and `bitsandbytes==0.49.2`, checks the GPU, downloads the data, runs the CPU tests and runs the pilot.
4. After the pilot, choose `DEV_MODEL` with the team, give each teammate a `SHARD`, and run the dev-run cell. It loops in chunks of 50 questions and syncs to Drive after each one. If the session dies, run all cells again and it resumes.
5. Copy the result files into `results/` in the repo and commit them.

On Colab the same notebook mounts Drive and writes results there directly.

## Reproducing the results

All runtimes below were measured with SLURM accounting (`sacct`) on TC1 (one Tesla V100-PCIE-32GB) or on a laptop CPU. The model weights download on the first run of each model, which is included in the pilot times.

### 1. Dependencies
- **GPU (TC1):** Python 3.11 in a **uv venv**, not conda; conda breaks on TC1's network home directory. Set it up once on the head node (installing only):
  ```bash
  module load python/3.11 uv
  export UV_CACHE_DIR=/tmp/$USER-uvcache UV_LINK_MODE=copy
  uv venv --python "$(which python3.11)" ~/venvs/vlmsyc && source ~/venvs/vlmsyc/bin/activate
  uv pip install -r requirements-gpu.txt --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu124
  ```
  `requirements-gpu.txt` pins every version: torch 2.6.0+cu124, transformers 4.57.6, accelerate 1.10.1, bitsandbytes 0.49.2, numpy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2, and the rest. Each result file's `_meta` header also records the versions used.
- **CPU (laptop; tests, gate analysis, figures):** `python3 -m venv .venv && ./.venv/bin/pip install -r requirements-cpu.txt`, then `./.venv/bin/python -m pytest -q`. That runs 71 tests in about 5 s, with no GPU and no model weights (a mock model is used). `requirements-cpu.txt` pins the same versions as the GPU environment.

### 2. Data preparation
```bash
python scripts/download_data.py   # POPE json (pinned commit) + the 500 COCO val2014 images it uses (~80 MB)
python scripts/build_splits.py    # rebuilds data/splits.json; a test checks it equals the committed file
```
The split is by image: 200 dev images (600 questions) and 300 test images (900 questions), seed 0. If `data/` is somewhere else, set `DATA_DIR`.

### 3. Commands per phase (run from the repo root; GPU steps through SLURM on TC1)
| Phase | Command | What it produces | Measured runtime |
|---|---|---|---|
| 0 GPU check | `sbatch slurm/job_smoke.sh` | log only | 1 min 17 s |
| 3 Pilot | `sbatch slurm/job_pilot.sh` (both models, ± suffix) | `results/pilot.csv`, `results/pilot_dev_*.jsonl` | 9 min 29 s, incl. weight download; LLaVA fp16 alone 4 min 2 s |
| 4–5 Dev baseline + signals | `sbatch slurm/job_dev.sh` | `results/baseline_dev_*.jsonl`, `turn1_degraded_dev_*`, `signals_dev_*`, `signals_auroc_dev.csv` | 35 min 16 s (both models) |
| 5 Gates on dev (CPU) | `python scripts/fit_gates_dev.py` then `python scripts/make_figures.py` | `results/gates_dev.csv`, `h2_dev.csv`, `h3_*_dev.csv`, `sensitivity_pi_dev.csv`, `signals_by_a1_dev.csv`, **`frozen_gates.json`**, `figures/*_dev.png` | about 40 s on a laptop CPU |
| 6 Prompt mitigations on dev | `sbatch slurm/job_phase6.sh`, then `python scripts/score_phase6_dev.py` (CPU) | `results/{grounding,relook,nosuffix}_dev_*.jsonl`, `phase6_dev.csv` | 1 h 57 min 29 s (both models, 3 conditions) |
| 7 Dry runs (no test data) | CPU: `python scripts/run_final.py --dry-run-dev --resume`; GPU: `MODE=dev bash slurm/submit_final.sh` | `results/final_dryrun_dev/`, `figures/final_dryrun_dev/` | CPU about 35 s; GPU on 30 questions/model: 2 min 10 s + 4 min 49 s |
| 7 **Test run (once)** | `bash slurm/submit_final.sh` from a fresh clone at tag `pre-final` (see "Phase 7" above) | `results/*_test_*.jsonl`, `results/final_test/`, `figures/final_test/` | measured with `sacct`: job A 73424 (base + signals, both models) **59 min 00 s** (11 Nov 13:09:37–14:08:37); job B 73425 (grounding, re-look, analysis) **2 h 04 min 46 s** (14:08:38–16:13:24); both COMPLETED, exit 0 |
| 8 Failure gallery | `python scripts/make_failure_gallery.py --partition test` (pre-registered); `--a4-figures-only` for the 16 cm layout; `python scripts/make_failure_gallery_b.py --partition test` (post-hoc view B) | `results/failure_gallery_*.csv`, `failure_counts_*.csv`, `failure_view_b_test.csv`, `figures/failure_gallery_*`, `figures/failure_view_b_*` | a few seconds |
| 8 Descriptive test tables | `python scripts/describe_final.py --partition test` (Amendment C) | `results/describe_test/` | about 10 s |
| 9 Submission ZIP | `bash scripts/make_zip.sh` | `dist/sc4001_code_<commit>.zip` (tracked files only; no data, PDFs, weights, caches or logs) | a few seconds |

### 4. The test split runs once
- **Lock:** `src/vlmsyc/splits.py` refuses to load the test split unless `ALLOW_TEST=1`. Only `scripts/run_final.py --final` sets it, and only after its guards pass:
  - HEAD carries the tag `pre-final` and the tree is clean;
  - the frozen gates are applied as saved, never refit;
  - an existing output is never overwritten;
  - `--resume` requires every header line to match.
- **Tested:** a unit test scans every other script and module to make sure none can unlock the test split.
- **Decided on dev only:** all thresholds, gates and decision rules were fixed on dev before the tag (`results/analysis_plan.md`, Amendments A and B, Note C).
- **Analysis failures:** if the analysis script fails after the model runs, Note C allows fixing and re-running it on the saved outputs. No model inference is re-run and no rule changes; the fix is disclosed.
