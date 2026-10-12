"""Report: Reduces the per-sequence score CSVs from evaluate.py into the final metrics, split by single / multi chain"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ["single", "multi"]


# inlined from src.utils, which imports esm; this script only needs pandas
def make_logger(path, overwrite=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w" if overwrite else "a", buffering=1)

    def log(msg):
        print(msg)
        f.write(msg + "\n")
        f.flush()
    return log, f


def ensure_output_paths(path):
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------
# Mode labelling
# ------------------
def build_mode_map(single_pdb_dir, multi_pdb_dir):
    """pdb name -> 'single' | 'multi', taken from which test folder holds the .pdb"""
    mode_map = {}
    for mode, folder in zip(MODES, [single_pdb_dir, multi_pdb_dir]):
        if folder is None:
            continue
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"[Error] PDB folder not found: {folder}")
        for f in os.listdir(folder):
            if not f.endswith(".pdb"):
                continue
            name = f[:-4]
            if mode_map.get(name, mode) != mode:
                raise ValueError(f"[Error] {name}.pdb is in both single and multi folders")
            mode_map[name] = mode
    if not mode_map:
        raise ValueError("[Error] No .pdb files found in the given folders")
    return mode_map


def label_modes(df, mode_map, log):
    pdb_names = df["Sequence_ID"].astype(str).str.split("_").str[0]
    df = df.copy()
    df["chain_mode"] = pdb_names.map(mode_map)

    n_missing = int(df["chain_mode"].isna().sum())
    if n_missing:
        missing = df.loc[df["chain_mode"].isna(), "Sequence_ID"].tolist()
        log(f"[Warning] {n_missing} rows have no matching PDB in either folder, dropped: {missing[:10]}")
        df = df.dropna(subset=["chain_mode"]).reset_index(drop=True)
    return df


def add_diff_count(df, log):
    """Number of critical positions, recovered from the stored GT / negative pair"""
    gt = df["GT_sequence"].astype(str)
    neg = df["Negative_sequence"].astype(str)

    bad_len = (gt.str.len() != neg.str.len())
    if bad_len.any():
        log(f"[Warning] {int(bad_len.sum())} rows have GT/negative length mismatch, diff_count set to NaN")

    counts = [np.nan if bad else sum(a != b for a, b in zip(g, n))
              for g, n, bad in zip(gt, neg, bad_len)]

    df = df.copy()
    df["diff_count"] = counts
    return df


# ------------------
# Metrics
# ------------------
def summarize_group(g):
    """Reduce per-sequence rows into the summary metrics"""
    seq_recovery = g["rec_pos"].mean()
    critical_recovery = g["rec_diff_pos"].mean()
    critical_recovery_neg = g["rec_diff_neg"].mean()

    perplexity = np.exp(-g["ll_pos_avg"]).mean()  # ll_pos_avg is the per-residue LL

    pairwise_acc = (g["ll_pos_avg"] > g["ll_neg_avg"]).mean()  # does GT rank above the negative?
    pairwise_acc_sum = (g["ll_pos"] > g["ll_neg"]).mean()
    critical_acc = (g["ll_diff_pos_avg"] > g["ll_diff_neg_avg"]).mean()

    return pd.Series({
        "n": len(g),
        "diff_count_mean": g["diff_count"].mean(),
        "seq_recovery": seq_recovery,
        "critical_recovery": critical_recovery,
        "critical_recovery_neg": critical_recovery_neg,
        "perplexity": perplexity,
        "pairwise_acc": pairwise_acc,
        "pairwise_acc_sum_ll": pairwise_acc_sum,
        "critical_acc": critical_acc,
        "ll_pos_avg": g["ll_pos_avg"].mean(),
        "ll_neg_avg": g["ll_neg_avg"].mean(),
        "ll_margin": (g["ll_pos_avg"] - g["ll_neg_avg"]).mean(),
    })


def summarize(df):
    """Per (model, chain_mode) plus an 'all' row that pools both modes."""
    per_mode = df.groupby(["model", "chain_mode"], sort=True).apply(summarize_group).reset_index()

    pooled = df.groupby("model", sort=True).apply(summarize_group).reset_index()
    pooled.insert(1, "chain_mode", "all")

    out = pd.concat([per_mode, pooled], ignore_index=True)
    out["chain_mode"] = pd.Categorical(out["chain_mode"], categories=MODES + ["all"], ordered=True)
    out = out.sort_values(["model", "chain_mode"]).reset_index(drop=True)
    out["n"] = out["n"].astype(int)
    return out


# ------------------
# Plots
# ------------------
def plot_seq_recovery(models, values, path, plt):
    """ProteinMPNN Evaluation.py SR_barplot.png"""
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values, color="#4C72B0", width=0.6)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Average Sequence Recovery")
    ax.set_title("Average Sequence Recovery per Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_critical_recovery(plot_df, models, path, plt, sns, Line2D):
    """ProteinMPNN Evaluation.py critical_recovery.png"""
    sns.set_theme(style="whitegrid", font_scale=1.4)
    plt.figure(figsize=(16, 7), dpi=300)
    palette = sns.color_palette("Set2", len(models))
    sns.violinplot(data=plot_df, x="Model", y="Recovery (%)", hue="Model", order=models,
                   palette=palette, inner=None, linewidth=1.1, cut=0, legend=False)

    OFFSET = 2
    for i, label in enumerate(models):
        mean_val = plot_df.loc[plot_df["Model"] == label, "Recovery (%)"].mean()
        plt.hlines(mean_val, i - 0.25, i + 0.25, colors="black", linestyles="--", linewidth=2)
        plt.text(i, mean_val + OFFSET, f"{mean_val:.2f}%", ha="center", fontsize=12)

    mean_line = Line2D([0], [0], color="black", linestyle="--", lw=2)
    plt.legend(handles=[mean_line], labels=["Mean"], loc="upper right")

    plt.title("Critical Position Recovery", fontsize=18)
    plt.ylabel("Critical Sequence Recovery (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_accuracy(models, values, path, plt, np_):
    """ProteinMPNN Perplexity.py model_accuracy_barplot.png"""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np_.arange(len(models))
    colors = plt.cm.tab20(np_.linspace(0, 1, len(models)))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Accuracy (Average Positive Score < Average Negative Score)")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(np_.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}" for v in np_.linspace(0, 1, 6)])

    for bar, acc in zip(bars, values):
        if not np_.isnan(acc):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.015, f"{acc * 100:.2f}%",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_plots(df, summary, out_dir, log):
    """One set of ProteinMPNN-style plots per chain_mode"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.lines import Line2D
    except ImportError as e:
        log(f"[Warning] Plotting skipped, missing dependency: {e}")
        return

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    for mode in MODES:
        rows = df[df["chain_mode"] == mode]
        stats = summary[summary["chain_mode"] == mode]
        if rows.empty or stats.empty:
            continue
        models = stats["model"].tolist()

        for name, fn in [
            (f"SR_barplot_{mode}.png",
             lambda p: plot_seq_recovery(models, stats["seq_recovery"].values, p, plt)),
            (f"critical_recovery_{mode}.png",
             lambda p: plot_critical_recovery(
                 pd.DataFrame({"Recovery (%)": rows["rec_diff_pos"].values * 100,
                               "Model": rows["model"].values}), models, p, plt, sns, Line2D)),
            (f"model_accuracy_barplot_{mode}.png",
             lambda p: plot_accuracy(models, stats["pairwise_acc"].values, p, plt, np)),
        ]:
            path = Path(out_dir) / name
            fn(path)
            log(f"Saved plot: {path}")


# ------------------
# Main
# ------------------
def main(args):
    log, log_f = make_logger(args.log_file)
    out_dir = ensure_output_paths(args.out_dir)
    mode_map = build_mode_map(args.single_pdb_dir, args.multi_pdb_dir)
    log(f"Mode map built from PDB folders: {sum(v == 'single' for v in mode_map.values())} single, "
        f"{sum(v == 'multi' for v in mode_map.values())} multi")

    frames = []
    for csv_path in args.score_csv:
        df = pd.read_csv(csv_path)
        model = args.model_names.pop(0) if args.model_names else Path(csv_path).stem
        df.insert(0, "model", model)
        df = label_modes(df, mode_map, log)
        counts = df["chain_mode"].value_counts().to_dict()
        log(f"[{model}] {len(df)} rows from {csv_path} -> {counts}")
        frames.append(df)

    per_row = pd.concat(frames, ignore_index=True)

    required = ["GT_sequence", "Negative_sequence",
                "rec_pos", "rec_diff_pos", "rec_diff_neg", "ll_pos", "ll_neg",
                "ll_pos_avg", "ll_neg_avg", "ll_diff_pos_avg", "ll_diff_neg_avg"]
    missing = [c for c in required if c not in per_row.columns]
    if missing:
        raise ValueError(f"[Error] Score CSV is missing columns from evaluate.py: {missing}")

    per_row = add_diff_count(per_row, log)

    # few mutated positions -> critical recovery only takes a handful of discrete values
    if args.min_diff > 0:
        keep = per_row["diff_count"] >= args.min_diff
        log(f"Filtering diff_count >= {args.min_diff}: kept {int(keep.sum())} / {len(per_row)} rows")
        per_row = per_row[keep].reset_index(drop=True)
        if per_row.empty:
            raise ValueError(f"[Error] No rows left after --min_diff {args.min_diff}")

    summary = summarize(per_row)

    per_row_path = out_dir / "per_sequence_labelled.csv"
    summary_path = out_dir / "summary.csv"
    per_row.to_csv(per_row_path, index=False)
    summary.to_csv(summary_path, index=False)

    log("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"\n✅ File saved: {per_row_path}")
    log(f"✅ File saved: {summary_path}")

    if args.plot:
        make_plots(per_row, summary, out_dir, log)

    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report final ESM-IF metrics from per-sequence evaluation scores")
    # Data
    parser.add_argument("--score_csv", nargs="+", required=True,
                        help="One or more score CSVs produced by src.evaluate (one per model)")
    parser.add_argument("--model_names", nargs="*", default=None,
                        help="Optional display names, matched to --score_csv in order (default: filename stem)")
    parser.add_argument("--single_pdb_dir", type=str, default="data/pdb/test_single_chain",
                        help="Folder of single-chain test PDBs, used to label rows")
    parser.add_argument("--multi_pdb_dir", type=str, default="data/pdb/test_multi_chain",
                        help="Folder of multi-chain test PDBs, used to label rows")
    parser.add_argument("--min_diff", type=int, default=0,
                        help="Drop sequences with fewer than this many mutated positions (0 keeps everything)")

    # I/O
    parser.add_argument("--out_dir", type=str, default="./results/summary",
                        help="Path to save the labelled per-sequence CSV, the summary CSV and plots")
    parser.add_argument("--log_file", type=str, default="./logs/report.log", help="Log file path")
    parser.add_argument("--plot", action="store_true", help="Also render bar / violin plots")

    args = parser.parse_args()
    main(args)
