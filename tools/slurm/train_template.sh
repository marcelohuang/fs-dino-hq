#!/bin/bash
# Generic Slurm template for one training command on one GPU. Adjust the
# #SBATCH lines and the environment setup to your cluster, then submit with
#   sbatch tools/slurm/train_template.sh train-stage2 --config configs/stage2_five_shot.json
# Training commands that support --resume (distill, train-stage2) continue
# from the last saved state when the job is resubmitted with --resume.
#SBATCH --job-name=fs-dino-hq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail

# Environment: activate the Python environment where fs-dino-hq is installed.
# source /path/to/venv/bin/activate

# Locations referenced by the configs as ${DATA_DIR}, ${WEIGHTS_DIR}, ${OUTPUT_DIR}.
: "${DATA_DIR:?set DATA_DIR}" "${WEIGHTS_DIR:?set WEIGHTS_DIR}" "${OUTPUT_DIR:?set OUTPUT_DIR}"
export DATA_DIR WEIGHTS_DIR OUTPUT_DIR

nvidia-smi -L || true
python -c "import torch, fs_dino_hq; print('torch', torch.__version__, 'fs_dino_hq', fs_dino_hq.__version__)"
fs-dino-hq "$@" --num-workers "${SLURM_CPUS_PER_TASK:-4}"
