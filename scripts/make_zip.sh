#!/bin/bash
# Build the submission ZIP from COMMITTED (tracked) files only, via `git archive`,
# so nothing untracked or uncommitted can slip in.
#
#   bash scripts/make_zip.sh                 # from the commit at HEAD
#   bash scripts/make_zip.sh <commit-or-tag>
#
# Included: everything tracked, including results/ and figures/.
# Excluded:
#   data/                       POPE files and COCO images (see README: data preparation), EXCEPT
#                               data/splits.json, which is included: it holds only the seed, the dev
#                               fraction, and image ids and question ids per partition (no labels, no images)
#   *.pdf                       e.g. the [RESTRICTED] TC1 user guide (also git-ignored, never committed)
#   checkpoints / weights       *.pt *.pth *.ckpt *.safetensors *.bin (none are tracked; excluded anyway)
#   caches                      __pycache__/ .pytest_cache/ *.pyc
#   logs                        results/logs_tc1/, *.out, *.err, *.log
# Guard: fails unless `git diff pre-final..<ref> -- src configs slurm` is empty (tagged code unchanged).
# Output: dist/sc4001_code_<short-commit>.zip (dist/ is git-ignored). Prints the file list and total size.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
REF="${1:-HEAD}"
COMMIT=$(git rev-parse --short "$REF")
if [ "$REF" = "HEAD" ] && [ -n "$(git status --porcelain)" ]; then
    echo "WARNING: uncommitted changes exist; they are NOT in the ZIP (it is built from $COMMIT)." >&2
fi
# The tagged code must be exactly what ran on test.
if ! git rev-parse -q --verify pre-final >/dev/null; then
    echo "FAILED: tag pre-final not found" >&2; exit 1
fi
if ! git diff --quiet pre-final "$REF" -- src configs slurm; then
    echo "FAILED: src/, configs/ or slurm/ differ from the pre-final tag:" >&2
    git diff --stat pre-final "$REF" -- src configs slurm >&2
    exit 1
fi
echo "OK: src/, configs/, slurm/ identical to pre-final"
# Every tracked file under data/ except splits.json is excluded (today only data/.gitkeep).
DATA_EXCLUDES=()
while IFS= read -r f; do
    [ "$f" = "data/splits.json" ] || DATA_EXCLUDES+=(":(exclude)$f")
done < <(git ls-tree -r --name-only "$REF" -- data)
mkdir -p dist
OUT="dist/sc4001_code_${COMMIT}.zip"
rm -f "$OUT"
git archive --format=zip --prefix="sc4001_project/" -o "$OUT" "$REF" -- . \
    "${DATA_EXCLUDES[@]}" \
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
DATA_IN_ZIP=$(unzip -Z1 "$OUT" | grep '^sc4001_project/data/.' || true)
echo "  data/ files: ${DATA_IN_ZIP:-none} (must be exactly sc4001_project/data/splits.json)"
[ "$DATA_IN_ZIP" = "sc4001_project/data/splits.json" ] || { echo "FAILED: unexpected data/ content" >&2; exit 1; }
for pat in '\.pdf$' '\.(pt|pth|ckpt|safetensors|bin)$' '__pycache__|\.pytest_cache|\.pyc$' 'logs_tc1/|\.(out|err|log)$'; do
    n=$(unzip -Z1 "$OUT" | grep -cE "$pat" || true)
    echo "  files matching $pat : $n (must be 0)"
    [ "$n" -eq 0 ] || { echo "FAILED: excluded files found" >&2; exit 1; }
done
echo "  results/ files: $(unzip -Z1 "$OUT" | grep -c '^sc4001_project/results/.' || true)"
echo "  figures/ files: $(unzip -Z1 "$OUT" | grep -c '^sc4001_project/figures/.' || true)"
echo
echo "== total: $(unzip -Z1 "$OUT" | grep -vc '/$') files, uncompressed $(unzip -l "$OUT" | tail -1 | awk '{printf "%.1f MB", $1/1e6}'), ZIP $(du -h "$OUT" | cut -f1)"
