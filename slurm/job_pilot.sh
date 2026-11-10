#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=pilot
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Phase 3 pilot on TC1. Submit from the repo root:  sbatch slurm/job_pilot.sh
# Runs the CPU tests, then 100 dev questions per model, with and without
# "Answer yes or no." appended to the pushback. Every run uses --resume, so
# resubmitting after a timeout continues where it stopped. Dev split only.
# Models to pilot can be overridden:  sbatch --export=ALL,MODELS="qwen2vl_2b" slurm/job_pilot.sh

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Refusing to run outside SLURM (no code on the head node). Use sbatch." >&2
    exit 1
fi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate mySC4001
cd "$SLURM_SUBMIT_DIR" || exit 1

# Model weights are cached in the home directory, so they download only once.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export RESULTS_DIR="${RESULTS_DIR:-$SLURM_SUBMIT_DIR/results}"
# The code is copied without .git, so the commit hash comes from a file.
if [ -f .git_commit ]; then export VLMSYC_GIT_COMMIT="$(cat .git_commit)"; fi
MODELS="${MODELS:-qwen2vl_2b llava15_7b_4bit}"

echo "job=$SLURM_JOB_ID node=$(hostname) date=$(date) commit=${VLMSYC_GIT_COMMIT:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m pytest -q tests || { echo "tests failed, stopping"; exit 1; }

for m in $MODELS; do
    echo "######## pilot: $m (instruction in turn 1 only) ########"
    python scripts/pilot.py --model-config "configs/models/$m.yaml" --resume
    echo "######## pilot: $m (+ 'Answer yes or no.' in the pushback) ########"
    python scripts/pilot.py --model-config "configs/models/$m.yaml" --resume \
        --pushback-suffix "Answer yes or no." --tag instr
done

echo "=== pilot.csv ==="
cat "$RESULTS_DIR/pilot.csv"
