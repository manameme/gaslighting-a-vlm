# When should a VLM accept a user's correction?

NTU SC4001 Neural Networks and Deep Learning, group project (Direction E: hallucination and reliability in VLMs).

A small open VLM answers a POPE yes/no question, the user pushes back, and we compare rules ("gates") that decide whether the model may change its answer. See `CLAUDE.md` for the study design and `SPEC.md` for the build spec (its "Changes" section lists the current scope).

**Status:** Phases 1–2 done (data, splits, model wrapper, two-turn pipeline, CPU tests). The pilot runs on Kaggle next. GPU runs use free Kaggle T4 GPUs because the NTU cluster could not be reached.

## Repo layout

| Path | Contents |
| --- | --- |
| `configs/` | `data.yaml`, `experiment.yaml`, `pushback.yaml`, one yaml per model in `models/` |
| `data/` | Downloaded POPE json and COCO images (not committed); `splits.json` (committed) |
| `src/vlmsyc/` | Library code: `data`, `splits`, `prompts`, `models`, `pipeline`, `metrics`, `paths` |
| `scripts/` | `download_data.py`, `build_splits.py`, `pilot.py`, `run_baseline.py`, `merge_shards.py`, `kaggle_runner.ipynb` |
| `tests/` | CPU-only tests using a mock model |
| `results/` | JSONL and CSV outputs, committed |
| `slurm/` | Unused (written for the NTU cluster before the move to Kaggle) |

## Where outputs go

All outputs go to `RESULTS_DIR` (environment variable, default `results/`). Inputs come from `DATA_DIR` (default `data/`).
Run scripts append rows and force them to disk after every batch, take `--resume` to skip finished rows, and take `--shard k/n` (k from 0) so that n people can each run a disjoint part. Each shard writes `baseline_dev_<model>_shard<k>of<n>.jsonl`. `scripts/merge_shards.py --model <model>` merges them into `baseline_dev_<model>.jsonl` and reports missing shards, duplicates and conflicts.

## Laptop setup (CPU tests only)

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy pyyaml pillow pytest
./.venv/bin/python -m pytest -q          # 43 tests, about 1 s
```

## Data preparation

```bash
python scripts/download_data.py   # POPE json at a pinned commit + the 500 COCO val2014 images it uses (~80 MB)
python scripts/build_splits.py    # only needed to rebuild data/splits.json (already committed)
```

- POPE: github.com/AoiDragon/POPE, commit `109985c74a00a88ba5341f9b0c1d4900f2a1f789`, MIT licence. We use the three COCO files (`random`, `popular`, `adversarial`).
- COCO val2014 images from images.cocodataset.org. COCO annotations are CC BY 4.0; the images come from Flickr under their own licence terms (see cocodataset.org/#termsofuse). Only the 500 images POPE references are downloaded.
- Split: 500 images → 200 dev / 300 test by image. Each image gives one question per POPE split, with yes/no balanced, so there are 500 questions per POPE split (dev 200 per split). See `results/splits_summary.csv`.

## Running on Kaggle

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

TODO (Phases 4 to 9). Include exact commands and runtimes measured in the pilot.
