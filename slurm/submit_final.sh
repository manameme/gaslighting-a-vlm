#!/bin/bash
# Submit the final run as two SLURM jobs: A (base + signals), then B (grounding,
# re-look, analysis), which starts only if A finishes successfully.
# Runs sbatch only (allowed on the head node); no Python here.
#
#   bash slurm/submit_final.sh                         # the real run (MODE=final)
#   MODE=dev OUT_DIR=$HOME/devtest bash slurm/submit_final.sh   # GPU rehearsal on 30 dev questions
set -euo pipefail
export MODE="${MODE:-final}"
if [ "$MODE" = "dev" ]; then
    export OUT_DIR="${OUT_DIR:-$HOME/devtest_$(date +%Y%m%d_%H%M%S)}"   # A and B must share it
    echo "dev rehearsal, outputs -> $OUT_DIR"
fi
A=$(sbatch --parsable --job-name="final${MODE:0:1}A" --export=ALL,PART=A slurm/job_final.sh)
B=$(sbatch --parsable --job-name="final${MODE:0:1}B" --dependency=afterok:"$A" --export=ALL,PART=B slurm/job_final.sh)
echo "submitted: A=$A  B=$B (B waits for A)"
squeue -u "$USER"
