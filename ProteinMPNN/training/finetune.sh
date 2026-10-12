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

set -euo pipefail

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/..")"         # .../ProteinMPNN
fi

TRAIN_CSV_PATH="${ROOT_PATH}/data/train_protein_mpnn_final.csv"
VAL_CSV_PATH="${ROOT_PATH}/data/valid_protein_mpnn_final.csv"
TRAIN_PDB_PATH="/scratch/prj0000000224/yinghua_proj/esm-if/data/train/"
VAL_PDB_PATH="/scratch/prj0000000224/yinghua_proj/esm-if/data/valid/"

DEBUG=False
NUM_NEIGHBORS=48
MODEL_NOISE=0.2
BACKBONE_NOISE=0.1
NUM_EPOCHS=20
SAVE_MODEL_EPOCH=1
ALPHA=0.2  # alpha is the lambda indicated in the paper: mask_sppo+alpha*ce (0 0.2 0.5 1 2)
BETA=0.2
BATCH_SIZE=10000

LOSS_TYPE="mask_sppo_ce"
if [ "$LOSS_TYPE" == "mask_sppo_ce" ]; then
    OUTPUT_PATH="final_results_/model_noise${MODEL_NOISE}/mask_sppo_ce_mn${MODEL_NOISE}_bn${BACKBONE_NOISE}_beta${BETA}_lambda${ALPHA}"
elif [ "$LOSS_TYPE" == "sppo" ]; then
    OUTPUT_PATH="final_results/model_noise${MODEL_NOISE}/baseline_sppo_mn${MODEL_NOISE}_bn${BACKBONE_NOISE}"
elif [ "$LOSS_TYPE" == "dpo" ]; then
    OUTPUT_PATH="final_results/model_noise${MODEL_NOISE}/baseline_dpo_mn${MODEL_NOISE}_bn${BACKBONE_NOISE}"
elif [ "$LOSS_TYPE" == "ce" ]; then
    OUTPUT_PATH="final_results/model_noise${MODEL_NOISE}/baseline_ce_mn${MODEL_NOISE}_bn${BACKBONE_NOISE}"
fi

echo $OUTPUT_PATH

python ./finetune.py \
        --debug $DEBUG \
        --train_pdb_path $TRAIN_PDB_PATH \
        --val_pdb_path $VAL_PDB_PATH \
        --train_csv_path $TRAIN_CSV_PATH \
        --val_csv_path $VAL_CSV_PATH \
        --model_noise $MODEL_NOISE \
        --backbone_noise $BACKBONE_NOISE \
        --num_neighbors 48 \
        --path_for_outputs $OUTPUT_PATH \
        --save_model_every_n_epochs $SAVE_MODEL_EPOCH \
        --num_epochs $NUM_EPOCHS \
        --alpha $ALPHA \
        --beta $BETA \
        --loss_type $LOSS_TYPE \
        --batch_size $BATCH_SIZE 
        
           