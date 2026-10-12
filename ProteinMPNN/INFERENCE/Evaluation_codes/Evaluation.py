#!/usr/bin/env python3

import os
import re
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import argparse


parser = argparse.ArgumentParser(description="ProteinMPNN Best Sample & Recovery Analysis")
parser.add_argument("--input_csv", type=str, required=True, help="CSV with sequence IDs, GT and negative sequences")
parser.add_argument("--inference_dir", type=str, required=True, help="Root folder containing model subfolders with seqs/")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSVs and plots")
parser.add_argument("--num_workers", type=int, default=max(1, cpu_count()-1), help="Number of parallel workers")
args = parser.parse_args()

INPUT_CSV = args.input_csv
ROOT_DIR = args.inference_dir
OUTPUT_DIR = args.output_dir
NUM_WORKERS = args.num_workers

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output paths
WIDE_CSV = os.path.join(OUTPUT_DIR, "best_seq_recovery_by_model.csv")
AVG_CSV  = os.path.join(OUTPUT_DIR, "model_average_summary.csv")
BEST_SAMPLES_CSV = os.path.join(OUTPUT_DIR, "best_samples.csv")
RECOVERY_CSV = os.path.join(OUTPUT_DIR, "best_samples_recovery.csv")
BAR_PNG = os.path.join(OUTPUT_DIR, "SR_barplot.png")
VIOLIN_PNG = os.path.join(OUTPUT_DIR, "critical_recovery.png")

# Regex
RE_SEQ_REC = re.compile(r"seq_recovery=([0-9.]+)")


# FUNCTION: Parse FASTA with seq_recovery

def parse_fasta_with_scores(filepath):
    entries = []
    with open(filepath) as f:
        block = []
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
        fullseq = "".join(entry[1:])
        m = RE_SEQ_REC.search(header)
        if not m:
            continue
        parsed.append({"recovery": float(m.group(1)), "seq": fullseq})
    return parsed


# FUNCTION: Get best sample for PDB

def get_best_sample(model_dir, pdb_id):
    best_sample, best_score = None, -1
    seqs_dir = os.path.join(model_dir, "seqs")
    if not os.path.isdir(seqs_dir):
        return None
    for file in os.listdir(seqs_dir):
        if file.lower().startswith(pdb_id.lower()) and file.endswith(".fa"):
            for s in parse_fasta_with_scores(os.path.join(seqs_dir, file)):
                if s["recovery"] > best_score:
                    best_score, best_sample = s["recovery"], s["seq"]
    return best_sample

def safe_seq(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\n","").replace("\r","").strip()


# STEP 1: Walk inference folder → best seq recovery per model

records = []
for root, dirs, files in os.walk(ROOT_DIR):
    if os.path.basename(root) != "seqs":
        continue
    model_name = os.path.basename(os.path.dirname(root))
    for fname in files:
        if not fname.endswith(".fa"):
            continue
        pdb_id = os.path.splitext(fname)[0].split("_")[0]
        seq_recoveries = [float(m.group(1)) for line in open(os.path.join(root, fname))
                          if (m:=RE_SEQ_REC.search(line))]
        if not seq_recoveries:
            continue
        records.append({"pdb_id": pdb_id, "model_name": model_name, "best_seq_recovery": max(seq_recoveries)})

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No valid seq_recovery found!")

# Wide table
df_wide = df.pivot(index="pdb_id", columns="model_name", values="best_seq_recovery")
df_wide.columns = [f"{c}_seq_recovery" for c in df_wide.columns]
df_wide = df_wide.reset_index()
df_wide.to_csv(WIDE_CSV, index=False)
print(f"Saved wide CSV: {WIDE_CSV}")

# Average per model
avg_records = [{"model_name": col.replace("_seq_recovery",""), "avg_seq_recovery": df_wide[col].mean(skipna=True)}
               for col in df_wide.columns if col.endswith("_seq_recovery")]
df_avg = pd.DataFrame(avg_records).sort_values("model_name").reset_index(drop=True)
df_avg.to_csv(AVG_CSV, index=False)
print(f"Saved model-average CSV: {AVG_CSV}")

# Bar plot
x = np.arange(len(df_avg))
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(x, df_avg["avg_seq_recovery"], color="#4C72B0", width=0.6)
for bar in bars:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(df_avg["model_name"], rotation=30, ha="right")
ax.set_ylabel("Average Sequence Recovery")
ax.set_title("Average Sequence Recovery per Model")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(BAR_PNG, dpi=300)
plt.close()
print(f"Saved bar plot: {BAR_PNG}")


# STEP 2: Best samples per sequence → recovery

# Detect models automatically
model_dirs = {name: os.path.join(ROOT_DIR, name) for name in os.listdir(ROOT_DIR)
              if os.path.isdir(os.path.join(ROOT_DIR, name))}
MODEL_LABELS = list(model_dirs.keys())
print(f"Detected models: {MODEL_LABELS}")

# Load input CSV
df_input = pd.read_csv(INPUT_CSV)
output_rows = []
for _, row in df_input.iterrows():
    seq_id, gt_seq, neg_seq = row["Sequence_ID"], safe_seq(row["GT_sequence"]), row.get("Neg_seq_ProteinMPNN_1","")
    pdb_id = seq_id.split("_")[0]
    model_best = {name: get_best_sample(dir, pdb_id) for name, dir in model_dirs.items()}
    record = {"Sequence_ID": seq_id, "GT_sequence": gt_seq, "Negative_sequence": neg_seq}
    record.update({f"{name}_best_sample": seq for name, seq in model_best.items()})
    output_rows.append(record)

output_df = pd.DataFrame(output_rows)
output_df.to_csv(BEST_SAMPLES_CSV, index=False)
print(f"Saved best samples CSV → {BEST_SAMPLES_CSV}")

# Recovery calculation
def compute_recovery(args):
    seq_id, gt, neg, *model_seqs = args
    L = min(len(gt), len(neg))
    diff_pos = [i for i in range(L) if gt[i]!=neg[i]]
    if not diff_pos:
        return {"Sequence_ID": seq_id, "diff_count": 0} | {f"{label}_recovery": np.nan for label in MODEL_LABELS}
    def rec(seq):
        if not seq: return np.nan
        return sum(1 for i in diff_pos if i<len(seq) and seq[i]==gt[i])/len(diff_pos)
    return {"Sequence_ID": seq_id, "diff_count": len(diff_pos)} | {f"{label}_recovery": rec(seq) for label, seq in zip(MODEL_LABELS, model_seqs)}

args_list = [[row["Sequence_ID"], safe_seq(row["GT_sequence"]), safe_seq(row["Negative_sequence"])] +
             [safe_seq(row[f"{label}_best_sample"]) for label in MODEL_LABELS] for _, row in output_df.iterrows()]

with Pool(NUM_WORKERS) as pool:
    results = pool.map(compute_recovery, args_list)

rec_df = pd.DataFrame(results)
merged_df = output_df.merge(rec_df, on="Sequence_ID", how="left")
merged_df.to_csv(RECOVERY_CSV, index=False)
print(f"Saved recovery CSV → {RECOVERY_CSV}")

# Violin plot
sns.set(style="whitegrid", font_scale=1.4)
plot_df = pd.DataFrame({
    "Recovery (%)": np.concatenate([merged_df[f"{label}_recovery"].dropna().values*100 for label in MODEL_LABELS]),
    "Model": np.concatenate([np.repeat(label, merged_df[f"{label}_recovery"].dropna().shape[0]) for label in MODEL_LABELS])
})

plt.figure(figsize=(16,7), dpi=300)
palette = sns.color_palette("Set2", len(MODEL_LABELS))
sns.violinplot(data=plot_df, x="Model", y="Recovery (%)", palette=palette, inner=None, linewidth=1.1, cut=0)

OFFSET = 2
for i, label in enumerate(MODEL_LABELS):
    vals = plot_df[plot_df["Model"]==label]["Recovery (%)"]
    mean_val = vals.mean()
    plt.hlines(mean_val, i-0.25, i+0.25, colors="black", linestyles="--", linewidth=2)
    plt.text(i, mean_val+OFFSET, f"{mean_val:.2f}%", ha="center", fontsize=12)

# Correct legend
mean_line = Line2D([0],[0], color="black", linestyle="--", lw=2)
plt.legend(handles=[mean_line], labels=["Mean"], loc="upper right")

plt.title("Critical Position Recovery", fontsize=18)
plt.ylabel("Critical Sequence Recovery (%)")
plt.ylim(0,100)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(VIOLIN_PNG, dpi=300)
plt.show()
print(f" Saved recovery plot → {VIOLIN_PNG}")
