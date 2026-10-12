#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." # ROOT

MODEL_PATH="checkpoints/run_20260806_142302/run_20260806_142302_epoch_039.pt"
TEMP=1e-6

python -m src.evaluate \
  --test_pdb_paths data/pdb/test_multi_chain data/pdb/test_single_chain \
  --test_csv_path data/test_esm-if_dataset_final.csv \
  --temperature "$TEMP" \
  --sample_n_seq 5 \
  --device cuda:0 \
  --model_path "$MODEL_PATH" \
  --out_dir results/evaluate \
  --log_file "logs/evaluate/$(basename "$MODEL_PATH" .pt)_T${TEMP}.log" \
  --save_logprob
