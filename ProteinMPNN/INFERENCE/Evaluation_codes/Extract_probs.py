#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast
import re
from datetime import datetime
from threading import Lock

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
NUM_WORKERS_NPZ = 4
NUM_WORKERS_FASTA = 16

lock = Lock()
log_entries = []
stats = {
    "npz_saved": 0,
    "npz_skipped": 0,
    "fasta_processed": 0,
    "fasta_skipped": 0,
    "errors": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract log_probs and split chains for ProteinMPNN")
    parser.add_argument("--base_dir", required=True, type=Path, help="Base directory for model run (contains probs, seqs, etc.)")
    parser.add_argument("--log_dir", required=True, type=Path, help="Directory for saving logs")
    return parser.parse_args()



def extract_log_probs(npz_file: Path, logprob_folder: Path):
    prefix = f"[NPZ:{npz_file.name}]"
    try:
        data = np.load(npz_file)
        if "log_probs" not in data:
            return f"{prefix} SKIP: no 'log_probs' key"

        log_probs = data["log_probs"]
        if log_probs.ndim < 2:
            return f"{prefix} ERROR: invalid log_probs shape {log_probs.shape}"

        # ProteinMPNN convention: select first model/ensemble
        log_probs = log_probs[0]

        if log_probs.ndim != 2:
            return f"{prefix} ERROR: unexpected shape after indexing {log_probs.shape}"

        if log_probs.shape[1] != len(AMINO_ACIDS):
            return f"{prefix} ERROR: expected {len(AMINO_ACIDS)} AA columns, got {log_probs.shape[1]}"

        df = pd.DataFrame(log_probs, columns=AMINO_ACIDS)
        out_csv = logprob_folder / f"{npz_file.stem}.csv"
        df.to_csv(out_csv, index=False)
        return f"{prefix} SAVED: {out_csv.name} shape={df.shape}"

    except Exception as e:
        return f"{prefix} EXCEPTION: {e}"


def run_npz_extraction(npz_folder: Path, logprob_folder: Path):
    npz_files = sorted(npz_folder.glob("*.npz"))
    with ProcessPoolExecutor(max_workers=NUM_WORKERS_NPZ) as executor:
        futures = {executor.submit(extract_log_probs, f, logprob_folder): f for f in npz_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="NPZ → log_probs CSV"):
            msg = future.result()
            with lock:
                log_entries.append(msg)
                if "SAVED" in msg:
                    stats["npz_saved"] += 1
                elif "SKIP" in msg:
                    stats["npz_skipped"] += 1
                elif "ERROR" in msg or "EXCEPTION" in msg:
                    stats["errors"] += 1



def process_fasta(fasta_file: Path, logprob_folder: Path, chain_csv_folder: Path):
    pdb_chain = fasta_file.stem
    prefix = f"[FASTA:{pdb_chain}]"

    try:
        with open(fasta_file) as f:
            header = f.readline().strip()
            seq_line = f.readline().strip()

        # Parse designed_chains
        match = re.search(r"designed_chains\s*=\s*(\[[^\]]*\])", header)
        if match:
            designed_chains = ast.literal_eval(match.group(1))
        else:
            chain_seqs = seq_line.split("/")
            designed_chains = [chr(65 + i) for i in range(len(chain_seqs))]

        chain_seqs = seq_line.split("/")
        if len(chain_seqs) != len(designed_chains):
            with lock:
                log_entries.append(f"{prefix} SKIP: chain count mismatch (chains={designed_chains}, seqs={len(chain_seqs)})")
                stats["fasta_skipped"] += 1
            return

        log_csv = logprob_folder / f"{pdb_chain}.csv"
        if not log_csv.exists():
            with lock:
                log_entries.append(f"{prefix} SKIP: missing log_probs CSV")
                stats["fasta_skipped"] += 1
            return

        df = pd.read_csv(log_csv)
        start = 0
        for chain_id, seq in zip(designed_chains, chain_seqs):
            L = len(seq)
            if start + L > len(df):
                with lock:
                    log_entries.append(f"{prefix} LENGTH MISMATCH: chain {chain_id} needs {L} rows, only {len(df)-start} left (total_rows={len(df)})")
                break
            out_df = df.iloc[start:start + L].copy()
            out_file = chain_csv_folder / f"{pdb_chain}_{chain_id}.csv"
            out_df.to_csv(out_file, index=False)
            start += L

        with lock:
            stats["fasta_processed"] += 1
            log_entries.append(f"{prefix} OK: chains={designed_chains}, total_seq_len={sum(len(s) for s in chain_seqs)}, log_rows={len(df)}")

    except Exception as e:
        with lock:
            log_entries.append(f"{prefix} EXCEPTION: {e}")
            stats["errors"] += 1


def run_fasta_processing(fasta_folder: Path, logprob_folder: Path, chain_csv_folder: Path):
    fasta_files = sorted(fasta_folder.glob("*.fa"))
    with ThreadPoolExecutor(max_workers=NUM_WORKERS_FASTA) as executor:
        futures = {executor.submit(process_fasta, f, logprob_folder, chain_csv_folder): f for f in fasta_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Splitting chains"):
            pass



def main():
    args = parse_args()

    BASE_DIR = args.base_dir
    NPZ_FOLDER = BASE_DIR / "probs"
    LOGPROB_CSV_FOLDER = BASE_DIR / "log_probs"
    CHAIN_CSV_FOLDER = BASE_DIR / "log_chain_probs"
    FASTA_FOLDER = BASE_DIR / "seqs"
    LOGPROB_CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    CHAIN_CSV_FOLDER.mkdir(parents=True, exist_ok=True)

    MASTER_LOG = args.log_dir / f"combined_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    run_npz_extraction(NPZ_FOLDER, LOGPROB_CSV_FOLDER)
    run_fasta_processing(FASTA_FOLDER, LOGPROB_CSV_FOLDER, CHAIN_CSV_FOLDER)

    summary = (
        "\n========== PIPELINE SUMMARY ==========\n"
        f"NPZ → CSV saved:      {stats['npz_saved']}\n"
        f"NPZ skipped:         {stats['npz_skipped']}\n"
        f"FASTA processed:     {stats['fasta_processed']}\n"
        f"FASTA skipped:       {stats['fasta_skipped']}\n"
        f"Errors:              {stats['errors']}\n"
    )

    with open(MASTER_LOG, "w") as f:
        f.write("\n".join(log_entries))
        f.write(summary)

    print(summary)
    print(f"Full log written to: {MASTER_LOG}")


if __name__ == "__main__":
    main()
