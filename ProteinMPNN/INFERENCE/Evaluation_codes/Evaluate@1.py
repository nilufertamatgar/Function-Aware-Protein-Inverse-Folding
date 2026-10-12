#!/usr/bin/env python3

import os
import re
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import argparse

# ===============================
# REPRODUCIBILITY
# ===============================
random.seed(42)
np.random.seed(42)

# ===============================
# ARGUMENT PARSER
# ===============================
parser = argparse.ArgumentParser(description="ProteinMPNN Random Sample & Recovery Analysis")
parser.add_argument("--input_csv", type=str, required=True,
                    help="CSV with Sequence_ID, GT_sequence, Negative sequence")
parser.add_argument("--inference_dir", type=str, required=True,
                    help="Root folder containing model subfolders with seqs/")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save CSVs and plots")
parser.add_argument("--num_workers", type=int, default=max(1, cpu_count()-1))
args = parser.parse_args()

INPUT_CSV = args.input_csv
ROOT_DIR = args.inference_dir
OUTPUT_DIR = args.output_dir
NUM_WORKERS = args.num_workers

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# OUTPUT PATHS
# ===============================
WIDE_CSV = os.path.join(OUTPUT_DIR, "random_seq_recovery_by_model.csv")
AVG_CSV  = os.path.join(OUTPUT_DIR, "model_average_summary.csv")
SAMPLES_CSV = os.path.join(OUTPUT_DIR, "random_samples.csv")
RECOVERY_CSV = os.path.join(OUTPUT_DIR, "random_samples_recovery.csv")
BAR_PNG = os.path.join(OUTPUT_DIR, "SR_barplot.png")
VIOLIN_PNG = os.path.join(OUTPUT_DIR, "critical_recovery.png")

# ===============================
# REGEX
# ===============================
RE_SEQ_REC = re.compile(r"seq_recovery=([0-9.]+)")

# ===============================
# FASTA PARSER
# ===============================
def parse_fasta_with_scores(filepath):
    entries, block = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                if block:
                    entries.append(block)
                block = [line.strip()]
            else:
                block.append(line.strip())
        if block:
            entries.append(block)

    parsed = []
    for entry in entries:
        header = entry[0]
        seq = "".join(entry[1:])
        m = RE_SEQ_REC.search(header)
        if m:
            parsed.append({
                "recovery": float(m.group(1)),
                "seq": seq
            })
    return parsed

# ===============================
# RANDOM SAMPLE FROM FASTA
# ===============================
def get_random_sample_and_recovery(model_dir, pdb_id):
    seqs_dir = os.path.join(model_dir, "seqs")
    if not os.path.isdir(seqs_dir):
        return None, None

    for fname in os.listdir(seqs_dir):
        if fname.lower().startswith(pdb_id.lower()) and fname.endswith(".fa"):
            entries = parse_fasta_with_scores(os.path.join(seqs_dir, fname))
            if not entries:
                return None, None
            chosen = random.choice(entries)
            return chosen["seq"], chosen["recovery"]

    return None, None

def safe_seq(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\n","").replace("\r","").strip()

# ===============================
# STEP 1: RANDOM seq_recovery PER FASTA
# ===============================
records = []

for root, _, files in os.walk(ROOT_DIR):
    if os.path.basename(root) != "seqs":
        continue

    model_name = os.path.basename(os.path.dirname(root))

    for fname in files:
        if not fname.endswith(".fa"):
            continue

        pdb_id = os.path.splitext(fname)[0].split("_")[0]
        entries = parse_fasta_with_scores(os.path.join(root, fname))
        if not entries:
            continue

        chosen = random.choice(entries)

        records.append({
            "pdb_id": pdb_id,
            "model_name": model_name,
            "random_seq_recovery": chosen["recovery"]
        })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No valid sequences found!")

# Wide table
df_wide = df.pivot(index="pdb_id", columns="model_name", values="random_seq_recovery")
df_wide = df_wide.reset_index()
df_wide.to_csv(WIDE_CSV, index=False)
print(f"✅ Saved random recovery CSV → {WIDE_CSV}")

# Model averages
avg_df = (
    df_wide
    .drop(columns=["pdb_id"])
    .mean(skipna=True)
    .reset_index()
    .rename(columns={"index": "model_name", 0: "avg_seq_recovery"})
)
avg_df.to_csv(AVG_CSV, index=False)
print(f"Saved model-average CSV → {AVG_CSV}")

# ===============================
# BAR PLOT
# ===============================
plt.figure(figsize=(8,5))
bars = plt.bar(avg_df["model_name"], avg_df["avg_seq_recovery"], color="#4C72B0")
for b in bars:
    plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
             ha="center", va="bottom", fontsize=9)
plt.ylabel("Average Sequence Recovery")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(BAR_PNG, dpi=300)
plt.close()
print(f"📊 Saved bar plot → {BAR_PNG}")

# ===============================
# STEP 2: RANDOM SAMPLE RECOVERY VS GT
# ===============================
model_dirs = {
    name: os.path.join(ROOT_DIR, name)
    for name in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, name))
}
MODEL_LABELS = list(model_dirs.keys())
print("Detected models:", MODEL_LABELS)

df_input = pd.read_csv(INPUT_CSV)

rows = []
for _, row in df_input.iterrows():
    pdb_id = row["Sequence_ID"].split("_")[0]
    record = {
        "Sequence_ID": row["Sequence_ID"],
        "GT_sequence": safe_seq(row["GT_sequence"]),
        "Negative_sequence": safe_seq(row.get("Neg_seq_ProteinMPNN_1", ""))
    }

    for name, d in model_dirs.items():
        seq, rec = get_random_sample_and_recovery(d, pdb_id)
        record[f"{name}_sample"] = seq
        record[f"{name}_seq_recovery"] = rec

    rows.append(record)

samples_df = pd.DataFrame(rows)
samples_df.to_csv(SAMPLES_CSV, index=False)
print(f"Saved random samples CSV → {SAMPLES_CSV}")

# ===============================
# CRITICAL POSITION RECOVERY

def compute_recovery(args):
    seq_id, gt, neg, *model_seqs = args
    L = min(len(gt), len(neg))
    diff_pos = [i for i in range(L) if gt[i] != neg[i]]

    if not diff_pos:
        return {"Sequence_ID": seq_id} | {f"{m}_recovery": np.nan for m in MODEL_LABELS}

    def rec(seq):
        if not seq:
            return np.nan
        return sum(seq[i] == gt[i] for i in diff_pos if i < len(seq)) / len(diff_pos)

    return {"Sequence_ID": seq_id} | {
        f"{m}_recovery": rec(s) for m, s in zip(MODEL_LABELS, model_seqs)
    }

args_list = [
    [r["Sequence_ID"], r["GT_sequence"], r["Negative_sequence"]] +
    [r[f"{m}_sample"] for m in MODEL_LABELS]
    for _, r in samples_df.iterrows()
]

with Pool(NUM_WORKERS) as pool:
    recs = pool.map(compute_recovery, args_list)

rec_df = pd.DataFrame(recs)
merged = samples_df.merge(rec_df, on="Sequence_ID")
merged.to_csv(RECOVERY_CSV, index=False)
print(f"Saved recovery CSV → {RECOVERY_CSV}")


plot_df = pd.DataFrame({
    "Recovery (%)": np.concatenate([
        merged[f"{m}_recovery"].dropna().values * 100 for m in MODEL_LABELS
    ]),
    "Model": np.concatenate([
        np.repeat(m, merged[f"{m}_recovery"].dropna().shape[0]) for m in MODEL_LABELS
    ])
})

plt.figure(figsize=(16,7), dpi=300)
sns.violinplot(data=plot_df, x="Model", y="Recovery (%)", inner=None, cut=0)

for i, m in enumerate(MODEL_LABELS):
    vals = plot_df[plot_df["Model"] == m]["Recovery (%)"]
    mean = vals.mean()
    plt.hlines(mean, i-0.25, i+0.25, colors="black", linestyles="--", lw=2)
    plt.text(i, mean+2, f"{mean:.1f}%", ha="center", fontsize=12)

plt.ylabel("Critical Sequence Recovery (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(VIOLIN_PNG, dpi=300)
plt.show()
print(f" Saved recovery plot → {VIOLIN_PNG}")
