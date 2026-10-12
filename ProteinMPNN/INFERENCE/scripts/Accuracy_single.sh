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


TEMP=0.1
SEED=13
POS_OR_NEG="Positive"  # Positive or Negative

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/../..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/../..")"         # .../ProteinMPNN
fi

PDB_DIR="${ROOT_PATH}/INFERENCE/Inputs/PDB_input/test_single_chain"
FASTA_DIR="${ROOT_PATH}/INFERENCE/Inputs/fasta_input/Single/${POS_OR_NEG}"

LOSS_TYPE="mask_sppo_ce"

if [[ "$LOSS_TYPE" == "mask_sppo_ce" ]]; then   
    MODEL_NOISE=0.2
    BACKBONE_NOISE=0.1
    LAMBDA=0.2
    MODEL_NAME="epoch_best"   # checkpoint name (without .pt)
    MODEL_WEIGHTS_DIR="${ROOT_PATH}/training/final_results/model_noise${MODEL_NOISE}/mask_sppo_ce_mn${MODEL_NOISE}_bn${BACKBONE_NOISE}_beta0.2_lambda${LAMBDA}/model_weights"
    Method_name="mn${MODEL_NOISE}_bn${BACKBONE_NOISE}_lambda${LAMBDA}_single"  #The name of the model like DPO, SPPO etc
    OUT_BASE="${ROOT_PATH}/INFERENCE/Scores_single/mn${MODEL_NOISE}/${Method_name}/${POS_OR_NEG}"
elif [[ "$LOSS_TYPE" == "base" ]]; then  
    BACKBONE_NOISE_="02"
    MODEL_NAME="v_48_020"   # checkpoint name (without .pt)
    MODEL_WEIGHTS_DIR="${ROOT_PATH}/vanilla_model_weights"
    Method_name="base"  #The name of the model like DPO, SPPO etc
    OUT_BASE="${ROOT_PATH}/INFERENCE/Scores_single/bn${BACKBONE_NOISE_}/${Method_name}/${POS_OR_NEG}"
else
    MODEL_NAME="epoch_best"   # checkpoint name (without .pt)
    MODEL_WEIGHTS_DIR="${ROOT_PATH}/training/final_results/baseline_${LOSS_TYPE}/model_weights"
    Method_name="baseline_${LOSS_TYPE}_single"  #The name of the model like DPO, SPPO etc
    OUT_BASE="${ROOT_PATH}/INFERENCE/Scores_single/baseline_${LOSS_TYPE}/${Method_name}/${POS_OR_NEG}"
fi


mkdir -p "$OUT_BASE"

echo "$ROOT_PATH"
echo "$MODEL_WEIGHTS_DIR"
echo "$OUT_BASE"


CHAINS_TO_DESIGN="A"


for pdb_path in "${PDB_DIR}"/*.pdb; do
    pdb_file=$(basename "$pdb_path")
    pdb_id="${pdb_file%.pdb}"

    echo "=============================="
    echo "Processing PDB: $pdb_id"
    echo "=============================="

    fasta_files=("${FASTA_DIR}/${pdb_id}"_*.fa)

    if [ ${#fasta_files[@]} -eq 0 ]; then
        echo " No FASTA files found for $pdb_id — skipping"
        continue
    fi

    for fasta_path in "${fasta_files[@]}"; do
        fasta_file=$(basename "$fasta_path")
        fasta_name="${fasta_file%.fa}"

        out_dir="${OUT_BASE}/${pdb_id}/${fasta_name}"
        mkdir -p "$out_dir"

        python ../../protein_mpnn_run.py \
            --path_to_model_weights "$MODEL_WEIGHTS_DIR" \
            --model_name "$MODEL_NAME" \
            --path_to_fasta "$fasta_path" \
            --pdb_path "$pdb_path" \
            --pdb_path_chains "$CHAINS_TO_DESIGN" \
            --out_folder "$out_dir" \
            --sampling_temp "$TEMP" \
            --num_seq_per_target 5 \
            --score_only 1 \
            --seed "$SEED" \
            --batch_size 1
    done
done

echo " All scoring jobs completed."
echo " Outputs saved under: $OUT_BASE"
