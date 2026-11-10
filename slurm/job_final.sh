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

# Phase 7: the ONE run on the test split, in two jobs (QoS: 1 GPU, so one at a time):
#   PART=A  base condition + signals (degraded passes, join), both models
#   PART=B  grounding + re-look, both models, then the frozen-gate analysis
# Submit both with slurm/submit_final.sh (B starts only after A succeeds).
#
# MODE=final (default): scripts/run_final.py --final. Must be submitted from a git
#   checkout of the commit tagged 'pre-final' with a clean tree (it refuses otherwise):
#     git clone ~/sc4001.bundle ~/final && cd ~/final && git checkout pre-final
# MODE=dev: the same two parts on the first LIMIT dev questions (default 30) with the
#   real models, written to OUT_DIR (default ~/devtest_<job>). No tag needed; for
#   rehearsing the final run on the GPU. Nothing here reads the test split.
#
# Data: POPE files and images come from DATA_DIR (default ~/project/data). run_final.py
# checks (in final mode) that DATA_DIR/splits.json equals the committed data/splits.json.
# After a timeout: resubmit the same part with RESUME=1.
# Expected time (final): A ~1.3 h, B ~2 h on a V100 (from the dev runs, x1.5 for 900 questions).

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Refusing to run outside SLURM (no code on the head node). Use sbatch." >&2
    exit 1
fi

module load python/3.11
source "$HOME/venvs/vlmsyc/bin/activate"
cd "$SLURM_SUBMIT_DIR" || exit 1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
unset VLMSYC_GIT_COMMIT RESULTS_DIR   # the commit comes from git; outputs go to this checkout's results/
export DATA_DIR="${DATA_DIR:-$HOME/project/data}"
MODE="${MODE:-final}"
PART="${PART:?set PART=A or PART=B (use slurm/submit_final.sh)}"

case "$PART" in
    A) STEPS="base" ;;
    B) STEPS="mitigations analysis" ;;
    *) echo "PART must be A or B" >&2; exit 1 ;;
esac

echo "job=$SLURM_JOB_ID part=$PART mode=$MODE node=$(hostname) date=$(date)"
echo "head=$(git rev-parse HEAD 2>/dev/null || echo 'no git') tags=$(git tag --points-at HEAD 2>/dev/null)"
echo "DATA_DIR=$DATA_DIR  (pope: $(ls "$DATA_DIR"/pope 2>/dev/null | wc -l) files, images: $(ls "$DATA_DIR"/coco_val2014 2>/dev/null | wc -l))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m pytest -q tests || { echo "tests failed, stopping"; exit 1; }

if [ "$MODE" = "final" ]; then
    python scripts/run_final.py --final --steps $STEPS ${RESUME:+--resume}
elif [ "$MODE" = "dev" ]; then
    OUT_DIR="${OUT_DIR:-$HOME/devtest_$SLURM_JOB_ID}"
    python scripts/run_final.py --dry-run-dev --limit "${LIMIT:-30}" --out-dir "$OUT_DIR" \
        --steps $STEPS ${RESUME:+--resume} --n-boot "${N_BOOT:-200}"
else
    echo "MODE must be final or dev" >&2; exit 1
fi
echo "done $(date)"
