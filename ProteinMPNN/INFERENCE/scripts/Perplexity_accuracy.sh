#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -J p-mpnn
#SBATCH -t 18:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH -p h200n
#SBATCH --mem-per-gpu=96G 

module load cuda/12.4.1
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate mlfold


MODE='single'  # 'single' or 'multi'

if [[ "$MODE" != "single" && "$MODE" != "multi" ]]; then
    echo "Usage: $0 [single|multi]"
    exit 1
fi

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/../..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/../..")"         # .../ProteinMPNN
fi

BASE_DIR="${ROOT_PATH}/INFERENCE/Scores_${MODE}/mn0.2"
#BASE_DIR="${ROOT_PATH}/INFERENCE/Scores_${MODE}/baseline_dpo"
OUTPUT_DIR="${ROOT_PATH}/INFERENCE/Results/mn0.2_${MODE}"

mkdir -p "$OUTPUT_DIR"


python "${ROOT_PATH}/INFERENCE/Evaluation_codes/Perplexity.py" "$BASE_DIR" "$OUTPUT_DIR"
#python "${ROOT_PATH}/INFERENCE/Evaluation_codes/Accuracy.py" "$BASE_DIR"
