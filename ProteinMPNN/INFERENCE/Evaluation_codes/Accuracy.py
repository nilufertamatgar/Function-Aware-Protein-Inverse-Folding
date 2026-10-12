#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GT vs Neg1 accuracy and generate bar plot"
    )

    parser.add_argument("--seq_csv", required=True, type=Path,
                        help="CSV containing Sequence_ID, GT_sequence, Neg_seq_ProteinMPNN_1")

    parser.add_argument("--log_dir_single", required=True, type=Path,
                        help="Directory with single-chain log probability CSVs")

    parser.add_argument("--log_dir_multi", required=True, type=Path,
                        help="Directory with multi-chain log probability CSVs")

    parser.add_argument("--out_csv_single", required=True, type=Path,
                        help="Output CSV for single-chain scores")

    parser.add_argument("--out_csv_multi", required=True, type=Path,
                        help="Output CSV for multi-chain scores")

    parser.add_argument("--out_plot", required=True, type=Path,
                        help="Output bar plot path")

    return parser.parse_args()




def score_sequence(seq, log_df):
    if len(seq) != len(log_df):
        raise ValueError("Sequence length != log matrix length")

    score = 0.0
    for i, aa in enumerate(seq):
        if aa not in log_df.columns:
            raise ValueError(f"Unknown amino acid: {aa}")
        score += log_df.iloc[i][aa]
    return score


def compute_accuracy(seq_csv, log_dir, out_csv, label):
    df = pd.read_csv(seq_csv)

    rows = []
    total = 0
    gt_wins = 0

    for _, row in df.iterrows():
        seq_id = row["Sequence_ID"]
        log_file = log_dir / f"{seq_id}.csv"

        gt_seq = row["GT_sequence"]
        neg_seq = row["Neg_seq_ProteinMPNN_1"]

        status = "ok"
        gt_score = np.nan
        neg_score = np.nan
        win = np.nan

        if not log_file.exists():
            status = "missing_log"
        elif len(gt_seq) != len(pd.read_csv(log_file)):
            status = "GT_length_mismatch"
        elif pd.isna(neg_seq):
            status = "Neg1_missing"
        elif len(neg_seq) != len(pd.read_csv(log_file)):
            status = "Neg1_length_mismatch"
        else:
            log_df = pd.read_csv(log_file)
            gt_score = score_sequence(gt_seq, log_df)
            neg_score = score_sequence(neg_seq, log_df)
            win = int(gt_score > neg_score)
            total += 1
            gt_wins += win

        rows.append({
            "Sequence_ID": seq_id,
            "GT_length": len(gt_seq),
            "Neg1_length": len(neg_seq) if pd.notna(neg_seq) else np.nan,
            "Log_rows": len(pd.read_csv(log_file)) if log_file.exists() else np.nan,
            "GT_score": gt_score,
            "Neg1_score": neg_score,
            "GT_gt_Neg1": win,
            "Status": status
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    accuracy = gt_wins / total if total > 0 else np.nan
    print(f"{label}: accuracy = {accuracy:.4f} ({gt_wins}/{total})")
    print(f"{label}: saved CSV → {out_csv}")

    return accuracy



def main():
    args = parse_args()

    acc_single = compute_accuracy(
        args.seq_csv,
        args.log_dir_single,
        args.out_csv_single,
        label="SINGLE"
    )

    acc_multi = compute_accuracy(
        args.seq_csv,
        args.log_dir_multi,
        args.out_csv_multi,
        label="MULTI"
    )

    labels = ["Single", "Multi"]
    accuracies = [acc_single, acc_multi]

    plt.figure()
    plt.bar(labels, accuracies)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy (GT > Neg1)")
    plt.title("GT vs Neg1 Accuracy: Single vs Multi")

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha="center")

    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_plot, dpi=300)
    plt.close()

    print(f"Saved accuracy bar plot to: {args.out_plot}")


if __name__ == "__main__":
    main()
