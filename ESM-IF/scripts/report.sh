#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." # ROOT

# Collects score CSVs written by scripts/evaluate.sh into one comparison table,
# split into single- / multi-chain by which test PDB folder holds each structure.
#   scripts/report.sh                 every CSV in results/evaluate/score_csv
#   scripts/report.sh a.csv b.csv     only these, in this order

SCORE_DIR="results/evaluate/score_csv"
OUT_DIR="results/summary"

if [ $# -gt 0 ]; then
    CSVS=("$@")
else
    shopt -s nullglob
    CSVS=("${SCORE_DIR}"/*.csv)
    shopt -u nullglob
fi

if [ ${#CSVS[@]} -eq 0 ]; then
    echo "No score CSVs found in ${SCORE_DIR} — run scripts/evaluate.sh first" >&2
    exit 1
fi

echo "Reporting on ${#CSVS[@]} model(s):"
printf '  %s\n' "${CSVS[@]}"

python -m src.report \
  --score_csv "${CSVS[@]}" \
  --single_pdb_dir data/pdb/test_single_chain \
  --multi_pdb_dir data/pdb/test_multi_chain \
  --out_dir "$OUT_DIR" \
  --log_file logs/report.log \
  --plot
