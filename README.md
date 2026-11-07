# When should a VLM accept a user's correction?

NTU SC4001 Neural Networks and Deep Learning, group project (Direction E: hallucination and reliability in VLMs).

A small open VLM answers a POPE yes/no question, the user pushes back, and we compare rules ("gates") that decide whether the model may change its answer. See `CLAUDE.md` for the study design and `SPEC.md` for the build spec.

**Status:** Phase 0 (setup). Sections marked TODO are filled in by later phases.

## Repo layout

| Path | Contents |
| --- | --- |
| `configs/` | One yaml per model, plus the experiment grid |
| `data/` | Not committed: POPE json and COCO images |
| `src/vlmsyc/` | Library code |
| `scripts/` | Entry points for each phase |
| `slurm/` | Job scripts for TC1 |
| `tests/` | CPU-only tests using a mock model |
| `results/` | JSONL and CSV outputs, committed |
| `figures/` | Figures for the report |
| `report/` | Report source |

## Setup on TC1

The cluster is reachable only from the NTU network, so connect to the NTU VPN when off campus. Replace `<username>` with your NTU account ID in lower case. Run Python and GPU commands only through `sbatch`, never on the head node.

**1. Upload the code from your laptop.** `.git` is included so that result rows can record the commit.

```bash
rsync -av --exclude='data/' --exclude='*.pdf' --exclude='.DS_Store' \
  ./ <username>@10.96.189.11:~/sc4001-vlm/
```

**2. Log in and check your limits.**

```bash
ssh <username>@10.96.189.11
MyTCinfo
```

**3. Create the conda environment.** This is needed once. If `conda init` changes your `.bashrc`, run `source ~/.bashrc` afterwards.

```bash
module load anaconda
conda init bash
source ~/.bashrc
cd ~
cp /tc1apps/3_conda_yml/SC4001.yml .
conda env create -f SC4001.yml -n mySC4001
conda activate mySC4001
```

**4. Add the project packages.** Read the plan conda prints before you accept it. PyTorch must not be changed or replaced.

```bash
conda install -c conda-forge --freeze-installed \
  transformers accelerate pillow pyyaml pandas scipy scikit-learn matplotlib pytest
```

**5. Run the smoke test.** It takes a few minutes once the job starts, plus any queue time. It loads no model and is safe to resubmit.

```bash
cd ~/sc4001-vlm
file slurm/job_smoke.sh
sbatch slurm/job_smoke.sh
squeue -u <username>
cat output_smoke_*.out error_smoke_*.err
```

The `file` command should report plain ASCII text. The last line of the output should read `required_checks=PASS`. The `can_reach_*` lines show whether compute nodes have internet access.

## Data preparation

TODO (Phase 1). Include dataset sources and licences for POPE and COCO val2014.

## Reproducing the results

TODO (Phases 3 to 9). Include exact commands and expected runtimes.

## Running the tests

TODO (Phase 1). Tests run on CPU with a mock model and never load real weights.
