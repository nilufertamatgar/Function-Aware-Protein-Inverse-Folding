#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -J p-mpnn
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH -p h200n
#SBATCH --mem-per-gpu=96G 

module load cuda/12.4.1
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate mlfold


BACKBONE_NOISE=0.2
MODE='single'

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/../..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/../..")"         # .../ProteinMPNN
fi

# Paths (default inputs/outputs)
INPUT_CSV="${ROOT_PATH}/data/test_protein_mpnn_final.csv"
INFERENCE_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL/mn${BACKBONE_NOISE}/${MODE}"
OUTPUT_DIR="${ROOT_PATH}/INFERENCE/Results/mn${BACKBONE_NOISE}_${MODE}"
## INFERENCE_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL_NEW/baseline/${MODE}"
## OUTPUT_DIR="${ROOT_PATH}/INFERENCE/Results_NEW/baseline_${MODE}"

echo "$INFERENCE_DIR"
echo "$OUTPUT_DIR"

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run combined analysis script
python "${ROOT_PATH}/INFERENCE/Evaluation_codes/Evaluation.py" \
    --input_csv "$INPUT_CSV" \
    --inference_dir "$INFERENCE_DIR" \
    --output_dir "$OUTPUT_DIR"
