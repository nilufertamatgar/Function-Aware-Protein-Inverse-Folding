#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -J p-mpnn
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH -p testqueue
#SBATCH --mem-per-gpu=96G 

module load cuda/12.4.1
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate mlfold

set -euo pipefail

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/../..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/../..")"         # .../ProteinMPNN
fi

# ---- Extract_probs.py inputs ----
BASE_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL/mn0.2/multi/mn0.2_bn0.1_lambda0.2_multi"
#BASE_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL_old/bn0.2/multi/base"

# ---- Accuracy.py inputs ----
SEQ_CSV="${ROOT_PATH}/data/test_protein_mpnn_final.csv"

LOG_DIR_SINGLE="${BASE_DIR}/log_chain_probs"
LOG_DIR_MULTI="${BASE_DIR}/log_chain_probs"

OUT_CSV_SINGLE="${ROOT_PATH}/INFERENCE/Pairwise_acc/mn0.2_bn0.1_lambda0.2_scores_single.csv"
OUT_CSV_MULTI="${ROOT_PATH}/INFERENCE/Pairwise_acc/mn0.2_bn0.1_lambda0.2_multi.csv"

OUT_PLOT="${ROOT_PATH}/INFERENCE/Pairwise_acc/Base_002_accuracy_bar.png"

# ---- Script locations ----
PYTHON_EXTRACT="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/Evaluation_codes/Extract_probs.py"
PYTHON_ACCURACY="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/Evaluation_codes/Accuracy.py"

# ---- Logging ----
LOG_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/logging"
RUN_LOG="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"



echo "========================================"
echo "ProteinMPNN pipeline started"
echo "BASE_DIR          : ${BASE_DIR}"
echo "SEQ_CSV           : ${SEQ_CSV}"
echo "LOG_DIR_SINGLE    : ${LOG_DIR_SINGLE}"
echo "LOG_DIR_MULTI     : ${LOG_DIR_MULTI}"
echo "OUT_PLOT          : ${OUT_PLOT}"
echo "========================================"



echo
echo ">>> STEP 1: Extract log_probs and split chains"
python "${PYTHON_EXTRACT}" \
    --base_dir "${BASE_DIR}" \
    --log_dir "${LOG_DIR}" \
    2>&1 | tee -a "${RUN_LOG}"

echo ">>> STEP 1 completed"



echo
echo ">>> STEP 2: Compute accuracy and generate plots"
python "${PYTHON_ACCURACY}" \
    --seq_csv "${SEQ_CSV}" \
    --log_dir_single "${LOG_DIR_SINGLE}" \
    --log_dir_multi "${LOG_DIR_MULTI}" \
    --out_csv_single "${OUT_CSV_SINGLE}" \
    --out_csv_multi "${OUT_CSV_MULTI}" \
    --out_plot "${OUT_PLOT}" \
    2>&1 | tee -a "${RUN_LOG}"

echo ">>> STEP 2 completed"



echo
echo "========================================"
echo "Pipeline finished successfully"
echo "Log file: ${RUN_LOG}"
echo "========================================"
