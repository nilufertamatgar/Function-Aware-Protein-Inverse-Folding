#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -J p-mpnn
#SBATCH -t 18:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH -p testqueue
#SBATCH --mem-per-gpu=96G 

module load cuda/12.4.1
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate mlfold

# Repo paths (prefer SLURM_SUBMIT_DIR, fall back to script location)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    ROOT_PATH="$(realpath "${SLURM_SUBMIT_DIR}/../..")"   # .../ProteinMPNN
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_PATH="$(realpath "${SCRIPT_DIR}/../..")"         # .../ProteinMPNN
fi

# Your directory here
#FA_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL_old/bn0.2/single/base/seqs"
#FA_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/single/mn0.2_bn0.1_lambda0.2_single/seqs"
FA_DIR="${ROOT_PATH}/INFERENCE/PDB_FINAL_old/bn0.2/multi/base/seqs"
#FA_DIR="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/multi/mn0.2_bn0.1_lambda0.2_multi/seqs"
# Optional arguments (use defaults if not provided)
MODEL="${1:-nferruz/ProtGPT2}"
PATTERN="${2:-*.fa}"
MODE="multi"

echo "=== Running ProtGPT2 Perplexity ==="
echo "Directory: $FA_DIR"
echo "Model: $MODEL"
echo "Pattern: $PATTERN"
echo "Mode: $MODE"
echo ""

# Check if directory exists
if [ ! -d "$FA_DIR" ]; then
    echo "ERROR: Directory not found: $FA_DIR"
    exit 1
fi

# Count files
FILE_COUNT=$(find "$FA_DIR" -name "$PATTERN" 2>/dev/null | wc -l)
if [ $FILE_COUNT -eq 0 ]; then
    echo "WARNING: No files found with pattern: $PATTERN"
    echo "Trying alternative patterns..."
    
    # Try common patterns
    for alt_pattern in "*.fa" "*.fasta" "*.fas"; do
        count=$(find "$FA_DIR" -name "$alt_pattern" 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "Found $count files with pattern: $alt_pattern"
            read -p "Use pattern '$alt_pattern' instead? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                PATTERN="$alt_pattern"
                FILE_COUNT=$count
                break
            fi
        fi
    done
    
    if [ $FILE_COUNT -eq 0 ]; then
        echo "No FASTA files found. Exiting."
        exit 1
    fi
fi

echo "Found $FILE_COUNT file(s)"
echo ""

# Run the Python script
python "${ROOT_PATH}/INFERENCE/Evaluation_codes/eval_protgpt.py" "$FA_DIR" --model "$MODEL" --pattern "$PATTERN" --mode "$MODE"
