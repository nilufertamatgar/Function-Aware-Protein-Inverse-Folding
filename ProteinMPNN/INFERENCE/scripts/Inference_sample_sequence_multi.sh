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

# -------------------------------
# INPUT DIRECTORIES
# -------------------------------

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/../..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/../..")"         # .../ProteinMPNN
fi

INPUT_PDB_DIR="${ROOT_PATH}/INFERENCE/Inputs/PDB_input/test_multi_chain"

# -------------------------------
# MODEL / OUTPUT SELECTION
# -------------------------------
# MODEL_NAME = checkpoint name (without .pt)
# OUTPUT_SUBDIR = folder name to save outputs (user-defined)
# MODEL_NAME="epoch_best" #.pt file name
# OUTPUT_SUBDIR="l01_bn0.2_multi"
# MODEL_NAME="v_48_002" #.pt file name
# OUTPUT_SUBDIR="base"
# MODEL_WEIGHTS_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/vanilla_model_weights"

LOSS_TYPE="mask_sppo_ce"

if [[ "$LOSS_TYPE" == "mask_sppo_ce" ]]; then
    # BACKBONE_NOISE=0.02
    # LAMBDA=0.2
    # MODEL_NAME="epoch_best"
    # OUTPUT_SUBDIR="bn${BACKBONE_NOISE}_lambda${LAMBDA}_multi"
    # OUTPUT_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/bn${BACKBONE_NOISE}/multi"
    # MODEL_WEIGHTS_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/training/final_results/backbone_noise${BACKBONE_NOISE}/mask_sppo_ce_beta0.2_lambda${LAMBDA}/model_weights"
    MODEL_NOISE=0.2
    BACKBONE_NOISE=0.1
    LAMBDA=0.2
    MODEL_NAME="epoch_best"
    OUTPUT_SUBDIR="mn${MODEL_NOISE}_bn${BACKBONE_NOISE}_lambda${LAMBDA}_multi"
    OUTPUT_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL/mn${MODEL_NOISE}/multi"
    MODEL_WEIGHTS_DIR="${ROOT_PATH}/training/final_results/model_noise${MODEL_NOISE}/mask_sppo_ce_mn${MODEL_NOISE}_bn${BACKBONE_NOISE}_beta0.2_lambda${LAMBDA}/model_weights"
else
    MODEL_NAME="epoch_best"
    OUTPUT_SUBDIR="baseline_${LOSS_TYPE}_multi"
    OUTPUT_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL/baseline/multi"
    MODEL_WEIGHTS_DIR="${ROOT_PATH}/training/final_results/baseline_${LOSS_TYPE}/model_weights"
fi



PT_FILE="${MODEL_WEIGHTS_DIR}/${MODEL_NAME}.pt"

if [ ! -f "$PT_FILE" ]; then
    echo "Checkpoint file not found: ${PT_FILE}"
    exit 1
fi

# Output directory per user-defined name
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_SUBDIR}"
mkdir -p "${MODEL_OUTPUT_DIR}"

# -------------------------------
# SAMPLING PARAMETERS
# -------------------------------
NUM_SEQ_PER_TARGET=5
SAMPLING_TEMP="0.1"
SEED=37
BATCH_SIZE=1

# -------------------------------
# PARSE CHAINS
# -------------------------------
PARSED_CHAINS_JSONL="${MODEL_OUTPUT_DIR}/parsed_pdbs.jsonl"
python ../../helper_scripts/parse_multiple_chains.py \
    --input_path "${INPUT_PDB_DIR}" \
    --output_path "${PARSED_CHAINS_JSONL}"

# -------------------------------
# RUN ProteinMPNN
# -------------------------------
python ../../protein_mpnn_run.py \
    --path_to_model_weights "${MODEL_WEIGHTS_DIR}" \
    --model_name "${MODEL_NAME}" \
    --jsonl_path "${PARSED_CHAINS_JSONL}" \
    --out_folder "${MODEL_OUTPUT_DIR}" \
    --num_seq_per_target "${NUM_SEQ_PER_TARGET}" \
    --sampling_temp "${SAMPLING_TEMP}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --save_probs 1

echo "Finished inference for checkpoint '${MODEL_NAME}', outputs saved in '${MODEL_OUTPUT_DIR}'"
