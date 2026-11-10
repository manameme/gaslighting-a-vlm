#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:30:00
#SBATCH --job-name=final
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Phase 7: the ONE run on the test split (scripts/run_final.py --final).
# Must be submitted from a git checkout of the commit tagged 'pre-final', with a
# clean tree (run_final.py refuses otherwise). Set-up on TC1 (see README):
#   git clone ~/sc4001.bundle ~/final && cd ~/final && git checkout pre-final
#   (data stays in ~/project/data; DATA_DIR points there, see below)
# First submission:     sbatch slurm/job_final.sh
# After a timeout:      sbatch --export=ALL,RESUME=1 slurm/job_final.sh
# Expected time on a V100 (from the dev runs, x1.5 for 900 questions): about 3 h
# for both models (Qwen ~1 h, LLaVA ~2 h; re-look is the slowest condition).

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Refusing to run outside SLURM (no code on the head node). Use sbatch." >&2
    exit 1
fi

module load python/3.11
source "$HOME/venvs/vlmsyc/bin/activate"
cd "$SLURM_SUBMIT_DIR" || exit 1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
unset VLMSYC_GIT_COMMIT RESULTS_DIR   # the commit comes from git; outputs go to this checkout's results/
# POPE files and images from the earlier copy; run_final.py checks that its splits.json
# is byte-identical to the committed data/splits.json.
export DATA_DIR="${DATA_DIR:-$HOME/project/data}"

echo "job=$SLURM_JOB_ID node=$(hostname) date=$(date) head=$(git rev-parse HEAD) tags=$(git tag --points-at HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m pytest -q tests || { echo "tests failed, stopping"; exit 1; }

python scripts/run_final.py --final ${RESUME:+--resume}
echo "done $(date)"
