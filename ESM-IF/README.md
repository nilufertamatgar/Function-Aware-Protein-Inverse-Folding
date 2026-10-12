# FPA on ESM-IF

This project extends the [ESM-IF repository](https://github.com/facebookresearch/esm/blob/main/examples/inverse_folding) with FPA fine-tuning and evaluation workflows.

Please follow the original repository for environment setup and basic usage instructions.

## Dataset
```bash
cd <repo_root>/ESM-IF

hf download yinghuayao/FPA-dataset \
  --repo-type dataset \
  --local-dir data \
  --include "*esm-if*" "ProteinMPNN_Eval_Inputs.tar.gz"

mkdir -p data/pdb
tar -xzf data/ProteinMPNN_Eval_Inputs.tar.gz -C data/pdb --strip-components=2 Inputs/PDB_input
```

The downloaded dataset contains the preference pairs used for FPA training and the PDB files required for evaluation.

After the commands above, `data/` should look like this:

```
data/
├── train_esm-if_dataset_final.csv     # training preference pairs
├── valid_esm-if_dataset_final.csv     # validation preference pairs
├── test_esm-if_dataset_final.csv      # test preference pairs
├── ProteinMPNN_Eval_Inputs.tar.gz
└── pdb/
    ├── test_single_chain/             # 834 PDB files
    └── test_multi_chain/              # 1452 PDB files
```

The PDB files above cover evaluation only. For training, download the corresponding structures from [RCSB PDB](https://www.rcsb.org/) in legacy `.pdb` format — the entry ID is the part of `Sequence_ID` before the underscore (e.g. `5nai_A` → `5nai`). Single entries are available at `https://files.rcsb.org/download/<pdb_id>.pdb`; for the full set, use the [RCSB batch download script](https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script).

Place them under `data/pdb/` as well, so that the training data sits next to the test data:

```
data/pdb/
├── train/
├── valid/
├── test_single_chain/
└── test_multi_chain/
```

Training and validation cover 36,298 and 2,552 unique entries respectively (~20GB unzipped). A small number of very large entries are only distributed as mmCIF; the loader expects `.pdb`, so drop those rows from the CSV or convert them.

The provided CSV files define the training/validation split during training, so no manual splitting is required. However, if you prefer to separate the PDB files into training and validation directories, follow the provided CSV splits.

## Pretrained FPA Checkpoint (Optional)

If you only want to reproduce the evaluation results or run inference, download the released FPA checkpoint:

```bash
cd <repo_root>/ESM-IF

hf download yinghuayao/FPA-model \
  esm-if_fpa_checkpoint.pt \
  --local-dir checkpoints
```

The checkpoint will be saved as:

```
checkpoints/
└── esm-if_fpa_checkpoint.pt
```

You can then set

```bash
MODEL_PATH="checkpoints/esm-if_fpa_checkpoint.pt"
```

in `scripts/evaluate.sh` or `scripts/inference.sh`.

## FPA Fine-Tuning

Open `scripts/train.sh` and set `--train_pdb_paths` and `--val_pdb_paths` to the directories where you placed the training PDB files. The remaining hyperparameters reproduce the setting reported in the paper. Then run:

```bash
bash scripts/train.sh
```

Checkpoints are written to `checkpoints/run_<timestamp>/`, and training logs to `logs/train/`.

## Evaluation

Evaluation runs in two steps: `scripts/evaluate.sh` scores the test set with one model, and `scripts/report.sh` turns those scores into the final metrics. Run the first once per model you want in the results table, then the second once.

### 1. Score the test set

Run the fine-tuning step above first and point the evaluation script at the checkpoint you produced. Open `scripts/evaluate.sh` and set `MODEL_PATH` to that checkpoint, for example:

```bash
MODEL_PATH="checkpoints/run_20260806_142302/run_20260806_142302_epoch_039.pt"
```

Leaving `MODEL_PATH` empty evaluates the pretrained ESM-IF baseline instead, which reproduces the baseline row of the results table.

```bash
bash scripts/evaluate.sh
```

This samples sequences for every test structure and writes one CSV of per-sequence scores to `results/evaluate/score_csv/`, named after the checkpoint. Repeat with a different `MODEL_PATH` for each model you want to compare; each run adds its own CSV.

### 2. Report the metrics

```bash
bash scripts/report.sh                                        # every model scored in step 1
bash scripts/report.sh results/evaluate/score_csv/a.csv ...   # only the given ones
```

Called without arguments it reads every CSV in `results/evaluate/score_csv/`, so all models scored in step 1 land in one table; pass paths to restrict it to a subset. Each row is one model and one subset, where the subset is `single` or `multi` depending on which folder holds that structure — `data/pdb/test_single_chain` or `data/pdb/test_multi_chain` — plus an `all` row pooling both.

The reported metrics are sequence recovery, critical-position recovery (over the positions where the ground-truth and negative sequences differ), perplexity of the ground-truth sequence, and pairwise preference accuracy.

Results are saved under `results/`:

```
results/
├── evaluate/
│   ├── score_csv/              # per-sequence scores, one CSV per model
│   ├── sampled_sequences/      # sequences sampled from each structure
│   └── log_probability/        # per-residue log-probability matrices (--save_logprob)
└── summary/
    ├── summary.csv             # the final metrics, per model and subset
    ├── per_sequence_labelled.csv
    └── *.png                   # bar and violin plots
```

## Inference

To design sequences for your own structures, open `scripts/inference.sh`, set `MODEL_PATH` as above, and set `--pdb_folder_path` to a directory of PDB files. Then run:

```bash
bash scripts/inference.sh
```

Sampled sequences are saved under `results/inference/sampled_sequences/`, and per-residue log-probabilities under `results/inference/log_probability/` (`.pt`) and `results/inference/log_probability_csv/` (`.csv`).

To redesign only part of a structure, pass `--partial_seq_path <dir>`, where `<dir>` contains one `<pdb_name>.csv` per structure listing the residue indices to redesign; all other positions are kept fixed. Partial redesign requires a temperature greater than 0.