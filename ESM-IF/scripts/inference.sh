#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." # ROOT

MODEL_PATH="checkpoints/run_20260806_142302/run_20260806_142302_epoch_039.pt"
TEMP=1e-6

# (Optional) Add `--partial_seq_path <dir>` to redesign only the positions listed in <dir>/<pdb_name>.csv (temperature > 0)
python -m src.inference \
  --pdb_folder_path data/pdb/<sample_pdb_path> \
  --temperature "$TEMP" \
  --sample_n_seq 5 \
  --device cuda:0 \
  --model_path "$MODEL_PATH" \
  --out_dir results/inference \
  --log_file "logs/inference/$(basename "$MODEL_PATH" .pt)_T${TEMP}.log" \
  --save_logprob
