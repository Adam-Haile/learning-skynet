#!/bin/bash

#SBATCH --job-name=muzero-training
#SBATCH --output=muzero-%j.out
#SBATCH --error=muzero-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300

# Stop if a command fails.
set -euo pipefail

# Print job information.
echo "Starting MuZero training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Move into the project directory.
cd "$SLURM_SUBMIT_DIR"
if [[ -f "train.py" && -f "__init__.py" ]]; then
    cd ..
fi
if [[ ! -f "MuZero/train.py" ]]; then
    echo "Cannot locate MuZero/train.py from submission directory" >&2
    exit 1
fi

# Activate the Python environment.
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Display GPU information.
nvidia-smi

# Run training.
args=(--require-gpu)
if [[ "${MUZERO_SMOKE_TEST:-0}" == "1" ]]; then
    args+=(--smoke-test)
fi
if [[ -n "${MUZERO_RESUME:-}" ]]; then
    args+=(--resume "$MUZERO_RESUME")
fi

# exec allows Slurm's pre-timeout USR1 to reach the Python process.
exec python -u -m MuZero.train_muzero "${args[@]}"

echo "MuZero training completed"
