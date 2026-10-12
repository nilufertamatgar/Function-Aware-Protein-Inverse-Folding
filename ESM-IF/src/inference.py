"""Sampling from PDB: takes a PDB folder, samples n sequences for every chain"""

import argparse, os
import glob
from pathlib import Path
from tqdm import tqdm

import pandas as pd

import torch

from src.utils import make_logger, ensure_output_paths, load_ckpt_inference, find_all_chains, get_coords_and_sequence
from src.model import sample_multiple_sequences

# ------------------
# Partial Seq (Fixed Positions)
# ------------------
def compute_partial_seq(folder_path, csv_name, native_seq):
    """preprocess fixed position using csv with masked position indices at column 0"""
    seq_aa = list(native_seq)

    csv_path = os.path.join(folder_path, f"{csv_name}.csv")
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    indices = df.iloc[:, 0].tolist()

    for idx in indices:
        if 0 <= idx < len(seq_aa):
            seq_aa[idx] = '<mask>'
           
    return seq_aa

# ------------------
# Main
# ------------------
def main(args):
    log, log_f = make_logger(args.log_file)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    model_name = Path(args.model_path).stem if args.model_path else "baseline"

    model, alphabet = load_ckpt_inference(args.model_path, model_name, device=device, log=log)
    model.eval()

    run_name = f"{model_name}_T{args.temperature}"
    seq_dir, log_prob_dir, log_prob_csv_dir = ensure_output_paths(f"{args.out_dir}/sampled_sequences", f"{args.out_dir}/log_probability/{run_name}", f"{args.out_dir}/log_probability_csv/{run_name}")

    seq_results = []

    pdb_files = os.path.join(args.pdb_folder_path, "**", "*.pdb")

    for pdb_path in tqdm(glob.glob(pdb_files, recursive=True), desc=f"Inference"):
        pdb_name = Path(pdb_path).stem
        all_chains = find_all_chains(pdb_path)

        for chain_id in all_chains:
            seq_id = f"{pdb_name}_{chain_id}"
            save_path = f"{log_prob_dir}/{seq_id}.pt" if args.save_logprob else None
            
            try:
                coords, native_seq = get_coords_and_sequence(pdb_path, chain_id, log)
            except Exception as e:
                log(f"[Skip] {seq_id} is skipped due to {e}")
                continue
            
            partial_seq = None
            if args.partial_seq_path:
                partial_seq = compute_partial_seq(args.partial_seq_path, pdb_name, native_seq)
                log(str(list(native_seq)))
                log(str(partial_seq))

            sampled_seq_list, sampled_tok_list, log_prob_list, idx = sample_multiple_sequences(model, coords, args.sample_n_seq, args.temperature, device, args.seed, save_path=save_path, positive_seq=native_seq, partial_seq=partial_seq)

            sampled_seq_dict = {f"Sampled_sequence_{i}": v for i, v in enumerate(sampled_seq_list)}

            seq_results.append({
                "Sequence_ID": seq_id,  
                "GT_sequence": native_seq,
                "Negative_sequence": "-",
                **sampled_seq_dict,
            })

            if args.save_logprob: 
                best_log_prob = log_prob_list[idx]
                log_prob_df = pd.DataFrame(best_log_prob.squeeze(0).T.cpu().detach().numpy(), columns=alphabet.all_toks)
                log_prob_df.to_csv(f"{log_prob_csv_dir}/{seq_id}.csv", index=False)


    seq_csv_path = f"{seq_dir}/{run_name}.csv"
    seq_df_out = pd.DataFrame(seq_results)
    seq_df_out.to_csv(seq_csv_path, index=False)
    
    log(f"✅ File saved: {seq_csv_path}")
    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM-IF Inference")
    # Data
    parser.add_argument("--pdb_folder_path", type=str, required=True, help="Folder containing PDB files to run inference on")
    parser.add_argument("--partial_seq_path", type=str, default="", help="(Optional) A folder containing pdb_name.csv with indices of mask position to run inference (partially fixed backbone)")

    # Sampling
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature")
    parser.add_argument("--sample_n_seq", type=int, default=5, help="Number of sequences to sample per structure")
    
    # System
    parser.add_argument("--device", type=str, required=True, help="cuda, cpu, or cuda:<index>")

    # I/O
    parser.add_argument("--model_path", type=str, default="", help="Path to checkpoint to load model, empty for the pretrained baseline")
    parser.add_argument("--out_dir", type=str, default="./inference/outputs", help="Path to save scores, sampled sequences and log-probability matrices (optional)")
    parser.add_argument("--log_file", type=str, default="./logs/inference.log", help="Log file path")
    parser.add_argument("--save_logprob", action="store_true", help="Save log-probability matrices for future evaluation")
    

    args = parser.parse_args()
    main(args)