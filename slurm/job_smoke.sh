#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:15:00
#SBATCH --job-name=smoke
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Phase 0 smoke test. Submit from the repo root:  sbatch slurm/job_smoke.sh
# Checks the GPU, torch, transformers and fp16, and records whether this
# compute node can reach the internet. It loads no model and downloads nothing.
# Exit code: 0 if the required checks pass. Blocked internet is a recorded
# result, not a failure, because we can upload the data by SFTP instead.

# Safety guard: the cluster forbids running Python or nvidia-smi on the head
# node, and SLURM_JOB_ID is only set inside a SLURM job.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Refusing to run outside SLURM. Use: sbatch slurm/job_smoke.sh" >&2
    exit 1
fi

# Conda fails on the NFS home (symlink loop), so we use a uv venv (see README).
module load python/3.11
source "$HOME/venvs/vlmsyc/bin/activate"

# Logs go to the folder sbatch was called from, so stay there.
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "=== JOB ==="
echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "date=$(date)"
echo "submit_dir=$SLURM_SUBMIT_DIR"
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"

echo "=== NVIDIA-SMI ==="
nvidia-smi || echo "nvidia-smi failed"

python - <<'PY'
"""Run each check separately so one failure does not hide the others."""
import importlib
import platform
import sys
import urllib.error
import urllib.request

failed = []  # names of required checks that did not pass
internet = {}  # host -> "yes" or "no"

print("=== PYTHON ===")
print("python", platform.python_version(), sys.executable)
if sys.version_info < (3, 10):
    failed.append("python>=3.10")

print("=== TORCH ===")
try:
    import torch

    print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("cudnn", torch.backends.cudnn.version())
    print("torch.cuda.is_available()", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu_memory_gb {total_gb:.1f}")
        # The project runs in fp16 on the V100, so do a real fp16 sum on the GPU.
        # An 8x8 matrix of ones times itself has entries of 8, which sum to 512.
        x = torch.ones(8, 8, dtype=torch.float16, device="cuda")
        fp16_sum = (x @ x).sum().item()
        print("fp16_matmul_sum", fp16_sum, "(expected 512.0)")
        if fp16_sum != 512.0:
            failed.append("fp16")
    else:
        failed.append("cuda")
except Exception as err:
    print("torch check FAILED:", repr(err))
    failed.append("torch")

print("=== TRANSFORMERS ===")
try:
    import transformers

    print("transformers", transformers.__version__)
    # transformers imports lazily, so name the classes we will need. A plain
    # "import transformers" can succeed even when these are broken.
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401

    print("vlm classes import OK")
except Exception as err:
    print("transformers check FAILED:", repr(err))
    failed.append("transformers")

print("=== OTHER PACKAGES (information only) ===")
for name in ["accelerate", "PIL", "yaml", "pandas", "scipy", "sklearn", "matplotlib", "pytest"]:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "version unknown"))
    except Exception as err:
        print(name, "MISSING:", repr(err))

print("=== INTERNET FROM COMPUTE NODE ===")
# These hosts serve the model weights, the POPE files and the COCO images.
# Any HTTP reply counts as reachable, even an error status such as 403.
hosts = {
    "huggingface": "https://huggingface.co",
    "github": "https://github.com",
    "coco_images": "http://images.cocodataset.org",
}
for name, url in hosts.items():
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=15) as reply:
            internet[name], detail = "yes", f"HTTP {reply.status}"
    except urllib.error.HTTPError as err:
        internet[name], detail = "yes", f"HTTP {err.code}"
    except Exception as err:
        internet[name], detail = "no", repr(err)
    print(name, internet[name], detail)

print("=== SUMMARY ===")
for name, reachable in internet.items():
    print(f"can_reach_{name}={reachable}")
print("required_checks=" + ("PASS" if not failed else "FAIL " + ",".join(failed)))
sys.exit(1 if failed else 0)
PY
