
import os
import pandas as pd
from pathlib import Path
import ast
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

seqs_folder = Path("/dapustor/nilufer/ProteinMPNN/INFERENCE/Final_v4_FINAL/Base_MPNN_020_multi/seqs")
log_csv_folder = Path("/dapustor/nilufer/ProteinMPNN/INFERENCE/Final_v4_FINAL/Base_MPNN_020_multi/log_probs")
output_folder = Path("/dapustor/nilufer/ProteinMPNN/INFERENCE/Final_v4_FINAL/Base_MPNN_020_multi/log_chain_probs")
main_log_folder = Path("/dapustor/nilufer/ProteinMPNN/INFERENCE/logging")

output_folder.mkdir(parents=True, exist_ok=True)
main_log_folder.mkdir(parents=True, exist_ok=True)

log_file = main_log_folder / f"chain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_entries = []
lock = Lock()
stats = {"processed": 0, "skipped": 0, "error": 0}


def process_fasta(fasta_file):
    pdb_chain = fasta_file.stem
    log_prefix = f"[{pdb_chain}]"

    try:
        with open(fasta_file, "r") as f:
            header = f.readline().strip()
            seq_line = f.readline().strip()

        # Extract designed_chains list from header
        match = re.search(r"designed_chains\s*=\s*(\[[^\]]*\])", header)
        if match:
            try:
                designed_chains = ast.literal_eval(match.group(1))
            except Exception as e:
                with lock:
                    log_entries.append(f"{log_prefix}  Failed to parse designed_chains: {e}")
                    stats["skipped"] += 1
                return
        else:
           
            chain_seqs = seq_line.split("/")
            designed_chains = [chr(65 + i) for i in range(len(chain_seqs))]

        chain_seqs = seq_line.split("/")
        if len(chain_seqs) != len(designed_chains):
            with lock:
                log_entries.append(f"{log_prefix}  Chain count mismatch (header vs sequences)")
                stats["skipped"] += 1
            return

        # Open the big log CSV
        log_csv_file = log_csv_folder / f"{pdb_chain}.csv"
        if not log_csv_file.exists():
            with lock:
                log_entries.append(f"{log_prefix}  Log CSV not found: {log_csv_file.name}")
                stats["skipped"] += 1
            return

        df = pd.read_csv(log_csv_file)

        # Slice rows for each chain and save
        start_row = 0
        for chain_id, seq in zip(designed_chains, chain_seqs):
            num_rows = len(seq)
            if start_row + num_rows > len(df):
                with lock:
                    log_entries.append(f"{log_prefix}  Row index out of range for chain {chain_id}")
                    stats["skipped"] += 1
                continue

            chain_df = df.iloc[start_row:start_row + num_rows].copy()
            out_file = output_folder / f"{pdb_chain}_{chain_id}.csv"
            chain_df.to_csv(out_file, index=False)
            start_row += num_rows

        with lock:
            stats["processed"] += 1
            log_entries.append(f"{log_prefix}  Processed {len(designed_chains)} chain(s)")

    except Exception as e:
        with lock:
            log_entries.append(f"{log_prefix}  ERROR: {e}")
            stats["error"] += 1



fasta_files = sorted(seqs_folder.glob("*.fa"))
max_workers = 16

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_fasta, f): f for f in fasta_files}
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing .fa files"):
        pass


summary = (
    f"\n--- PROCESSING SUMMARY ---\n"
    f" Chain CSVs written: {stats['processed']}\n"
    f" Skipped: {stats['skipped']}\n"
    f" Errors: {stats['error']}\n"
    f"Total .fa files scanned: {len(fasta_files)}\n"
)

with lock:
    log_entries.append(summary)
    with open(log_file, "w") as logf:
        logf.write("\n".join(log_entries))

print(summary)
print(f"Detailed log saved at: {log_file}")
