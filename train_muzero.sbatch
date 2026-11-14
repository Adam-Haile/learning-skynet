#!/bin/bash
#SBATCH --job-name=muzero
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.out
#
# Train the MuZero agent on Hidden Hand on one GPU (Tesla T4 on the teaching nodes).
#
#   mkdir -p runs/slurm && sbatch train_muzero.sbatch
#   sbatch train_muzero.sbatch --iterations 200 --seed 1     # extra args go to muzero.train
#   sbatch --gres=gpu:0 train_muzero.sbatch --device cpu      # CPU-only run
#
# Output lands in runs/muzero-<jobid>/ (history.json, model.pt, ckpt_*.pt) unless --out is passed.
# The first run creates .venv/ in the repo with CUDA PyTorch; later runs reuse it.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_DIR"

VENV="$REPO_DIR/.venv"
PYTHON_BASE="/usr/local/miniforge/miniforge3/bin/python3"   # Python 3.12
TORCH_INDEX="https://download.pytorch.org/whl/cu128"        # CUDA 12.8 wheels; the node driver is newer

if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import torch, hidden_hand, muzero; assert torch.version.cuda" 2>/dev/null; then
    echo "Setting up $VENV ..."
    "$PYTHON_BASE" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install --force-reinstall torch --index-url "$TORCH_INDEX"
    "$VENV/bin/pip" install -e ".[dev]"
fi

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1

OUT="runs/muzero-${SLURM_JOB_ID:-local}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | ${THREADS} CPUs | GPUs: ${CUDA_VISIBLE_DEVICES:-none} | git $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# --out is first so a user-supplied --out in "$@" overrides it. --device defaults to auto (cuda if present).
srun "$VENV/bin/python" -m muzero.train \
    --out "$OUT" \
    --threads "$THREADS" \
    "$@"
