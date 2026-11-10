#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=dev
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Phases 4-5 on the dev split, one model after another on one GPU:
#   1. baseline: 600 dev questions x 12 pushbacks  (scripts/run_baseline.py)
#   2. merge + completeness check                   (scripts/merge_shards.py)
#   3. signals: blurred/blank turn-1 passes, join, AUROC table (scripts/run_signals.py)
# Every step resumes, so after a timeout just submit the same command again.
# Submit from the repo root:  sbatch slurm/job_dev.sh
# Models can be overridden:   sbatch --export=ALL,MODELS="qwen2vl_2b" slurm/job_dev.sh
# Dev split only. The test split is read only by run_final.py --final.

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Refusing to run outside SLURM (no code on the head node). Use sbatch." >&2
    exit 1
fi

# Conda fails on the NFS home (symlink loop), so we use a uv venv (see README).
module load python/3.11
source "$HOME/venvs/vlmsyc/bin/activate"
cd "$SLURM_SUBMIT_DIR" || exit 1

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export RESULTS_DIR="${RESULTS_DIR:-$SLURM_SUBMIT_DIR/results}"
if [ -f .git_commit ]; then export VLMSYC_GIT_COMMIT="$(cat .git_commit)"; fi
MODELS="${MODELS:-qwen2vl_2b llava15_7b_fp16}"
set -e

echo "job=$SLURM_JOB_ID node=$(hostname) date=$(date) commit=${VLMSYC_GIT_COMMIT:-unknown} models=$MODELS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m pytest -q tests

for m in $MODELS; do
    cfg="configs/models/$m.yaml"
    echo "######## $m: baseline (dev) $(date +%T) ########"
    python scripts/run_baseline.py --model-config "$cfg" --resume
    echo "######## $m: merge $(date +%T) ########"
    python scripts/merge_shards.py --model "$m" --expect-questions 600 --expect-pushbacks 12
    echo "######## $m: signals (dev) $(date +%T) ########"
    python scripts/run_signals.py --model-config "$cfg" --resume
done
echo "done $(date)"
