#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# -----------------------------
# BASE DIRECTORY
# -----------------------------
if len(sys.argv) > 2:
    BASE_DIR = Path(sys.argv[1])
    OUTPUT_DIR = Path(sys.argv[2])
else:
    print("Required argument: BASE_DIR and OUTPUT_DIR")
    exit(1)

print(f"🔹 Using base directory: {BASE_DIR}")


# -----------------------------
# OUTPUT FILES
# -----------------------------
OUTPUT_MIN_CSV = OUTPUT_DIR / "avg_scores_combined_all_models.csv"
OUTPUT_AVG_CSV = OUTPUT_DIR / "avg_positive_perplexity_all_models.csv"
OUTPUT_ACC_CSV = OUTPUT_DIR / "model_accuracy.csv"
OUTPUT_ACC_PNG = OUTPUT_DIR / "model_accuracy_barplot.png"


# -----------------------------
# PLOT SETTINGS
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})


# -----------------------------
# HELPER FUNCTION
# -----------------------------
def get_top_folder(path, base_dir_name):
    parts = path.parts
    try:
        idx = parts.index(base_dir_name)
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


# -----------------------------
# FIND MODEL DIRECTORIES
# -----------------------------
model_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]

all_min_records = []
avg_results = []
acc_results = []

for model_dir in sorted(model_dirs):
    model_name = model_dir.name
    print(f"Processing model: {model_name}")

    pos_dir = model_dir / "Positive"
    neg_dir = model_dir / "Negative"

    # ONLY consider FASTA npz files, ignore pdb npz
    pos_files = [f for f in pos_dir.glob("**/score_only/*.npz") if "_fasta_" in f.name] if pos_dir.exists() else []
    neg_files = [f for f in neg_dir.glob("**/score_only/*.npz") if "_fasta_" in f.name] if neg_dir.exists() else []

    # Collect top-level folders
    all_folders = set(get_top_folder(f, "Positive") for f in pos_files)
    all_folders.update(get_top_folder(f, "Negative") for f in neg_files)

    # -----------------------------
    # AVERAGE SCORES PER FOLDER
    # -----------------------------
    for folder_name in sorted(all_folders):
        pos_folder_files = [f for f in pos_files if get_top_folder(f, "Positive") == folder_name]
        neg_folder_files = [f for f in neg_files if get_top_folder(f, "Negative") == folder_name]

        pos_score = np.mean([np.mean(np.load(f, allow_pickle=True)["score"]) for f in pos_folder_files]) if pos_folder_files else None
        neg_score = np.mean([np.mean(np.load(f, allow_pickle=True)["score"]) for f in neg_folder_files]) if neg_folder_files else None

        all_min_records.append({
            "model_name": model_name,
            "file_name": folder_name,
            "positive": pos_score,
            "negative": neg_score
        })

    # -----------------------------
    # AVERAGE POSITIVE PERPLEXITY
    # -----------------------------
    if pos_files:
        positive_scores = np.array([np.mean(np.load(f, allow_pickle=True)["score"]) for f in pos_files])
        avg_ppl = np.exp(positive_scores).mean()
        n_seq = len(positive_scores)
    else:
        avg_ppl = np.nan
        n_seq = 0

    avg_results.append({
        "model_name": model_name,
        "num_sequences": n_seq,
        "avg_positive_perplexity": avg_ppl
    })

    # -----------------------------
    # ACCURACY (positive < negative)
    # -----------------------------
    if pos_files and neg_files:
        df_temp = pd.DataFrame({
            "positive": [np.mean(np.load(f, allow_pickle=True)["score"]) for f in pos_files],
            "negative": [np.mean(np.load(f, allow_pickle=True)["score"]) for f in neg_files]
        }).dropna()

        if df_temp.empty:
            acc = np.nan
            n_acc = 0
        else:
            n_acc = len(df_temp)
            acc = (df_temp["positive"] < df_temp["negative"]).sum() / n_acc
    else:
        acc = np.nan
        n_acc = 0

    acc_results.append({
        "model_name": model_name,
        "accuracy": acc,
        "num_sequences": n_acc
    })


# -----------------------------
# SAVE CSVs
# -----------------------------
pd.DataFrame(all_min_records).to_csv(OUTPUT_MIN_CSV, index=False)
print(f"Saved avg scores CSV: {OUTPUT_MIN_CSV}")

pd.DataFrame(avg_results).sort_values("model_name").to_csv(OUTPUT_AVG_CSV, index=False)
print(f"Saved avg positive perplexity CSV: {OUTPUT_AVG_CSV}")

df_acc = pd.DataFrame(acc_results).sort_values("model_name")
df_acc.to_csv(OUTPUT_ACC_CSV, index=False)
print(f"Saved accuracy CSV: {OUTPUT_ACC_CSV}")


# -----------------------------
# PLOT ACCURACY
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(df_acc))
colors = plt.cm.tab20(np.linspace(0, 1, len(df_acc)))
bars = ax.bar(x, df_acc["accuracy"].values, color=colors, edgecolor="black", linewidth=0.6)

ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Model")
ax.set_title("Model Accuracy (Average Positive Score < Average Negative Score)")
ax.set_ylim(0, 1)
ax.set_xticks(x)
ax.set_xticklabels(df_acc["model_name"], rotation=30, ha="right")
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_yticklabels([f"{int(v*100)}" for v in np.linspace(0, 1, 6)])

for bar, acc in zip(bars, df_acc["accuracy"].values):
    if not np.isnan(acc):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.015, f"{acc*100:.2f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_ACC_PNG, dpi=300)
plt.close()
print(f"Saved accuracy bar plot to: {OUTPUT_ACC_PNG}")
