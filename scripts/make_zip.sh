#!/bin/bash
# Build the submission ZIP from COMMITTED (tracked) files only, via `git archive`,
# so nothing untracked or uncommitted can slip in.
#
#   bash scripts/make_zip.sh                 # from the commit at HEAD
#   bash scripts/make_zip.sh <commit-or-tag>
#
# Included: everything tracked, including results/ and figures/.
# Excluded:
#   data/                       POPE files, COCO images and splits.json (see README: data preparation)
#   *.pdf                       e.g. the [RESTRICTED] TC1 user guide (also git-ignored, never committed)
#   checkpoints / weights       *.pt *.pth *.ckpt *.safetensors *.bin (none are tracked; excluded anyway)
#   caches                      __pycache__/ .pytest_cache/ *.pyc
#   logs                        results/logs_tc1/, *.out, *.err, *.log
# Output: dist/sc4001_code_<short-commit>.zip (dist/ is git-ignored). Prints the file list and total size.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
REF="${1:-HEAD}"
COMMIT=$(git rev-parse --short "$REF")
if [ "$REF" = "HEAD" ] && [ -n "$(git status --porcelain)" ]; then
    echo "WARNING: uncommitted changes exist; they are NOT in the ZIP (it is built from $COMMIT)." >&2
fi
mkdir -p dist
OUT="dist/sc4001_code_${COMMIT}.zip"
rm -f "$OUT"
git archive --format=zip --prefix="sc4001_project/" -o "$OUT" "$REF" -- . \
    ':(exclude)data' \
    ':(exclude,glob)**/*.pdf' \
    ':(exclude,glob)**/*.pt' ':(exclude,glob)**/*.pth' ':(exclude,glob)**/*.ckpt' \
    ':(exclude,glob)**/*.safetensors' ':(exclude,glob)**/*.bin' \
    ':(exclude,glob)**/__pycache__/**' ':(exclude,glob)**/.pytest_cache/**' ':(exclude,glob)**/*.pyc' \
    ':(exclude)results/logs_tc1' \
    ':(exclude,glob)**/*.out' ':(exclude,glob)**/*.err' ':(exclude,glob)**/*.log'

echo "== files in $OUT (from commit $COMMIT):"
unzip -Z1 "$OUT" | grep -v '/$'
echo
echo "== checks:"
for pat in '^sc4001_project/data/' '\.pdf$' '\.(pt|pth|ckpt|safetensors|bin)$' '__pycache__|\.pytest_cache|\.pyc$' 'logs_tc1/|\.(out|err|log)$'; do
    n=$(unzip -Z1 "$OUT" | grep -cE "$pat" || true)
    echo "  files matching $pat : $n (must be 0)"
    [ "$n" -eq 0 ] || { echo "FAILED: excluded files found" >&2; exit 1; }
done
echo "  results/ files: $(unzip -Z1 "$OUT" | grep -c '^sc4001_project/results/.' || true)"
echo "  figures/ files: $(unzip -Z1 "$OUT" | grep -c '^sc4001_project/figures/.' || true)"
echo
echo "== total: $(unzip -Z1 "$OUT" | grep -vc '/$') files, uncompressed $(unzip -l "$OUT" | tail -1 | awk '{printf "%.1f MB", $1/1e6}'), ZIP $(du -h "$OUT" | cut -f1)"
