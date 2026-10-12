#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." # ROOT

RUN="run_$(date +%Y%m%d_%H%M%S)"

python -m src.train \
  --train_pdb_paths data/pdb/train \
  --val_pdb_paths data/pdb/valid \
  --train_csv_path data/train_esm-if_dataset_final.csv \
  --val_csv_path data/valid_esm-if_dataset_final.csv \
  --epochs 40 \
  --lr 1e-7 \
  --loss_name fpa \
  --loss_avg \
  --beta 0.1 \
  --lambda_ce 1.0 \
  --device cuda:0 \
  --save_dir "checkpoints/$RUN" \
  --save_every 5 \
  --save_best \
  --log_file "logs/train/$RUN.log"