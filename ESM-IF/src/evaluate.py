"""Sample & Score: Takes the test CSV with GT/negative sequences"""

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from esm.inverse_folding.util import CoordBatchConverter

from src.utils import make_logger, ensure_output_paths, load_ckpt_inference
from src.dataset import ESMIFPreferenceDataset, collate_fn
from src.model import prepare_input, sample_multiple_sequences

# ------------------
# Score Functions
# ------------------
def targets_token(model, coords, positive_seq, negative_seq, alphabet, batch_converter, device):
    """Tokenize a pos/neg pair: returns their targets and the non-padding / critical site masks"""
    coords = [coords]
    positive_seq = [positive_seq]
    negative_seq = [negative_seq]

    *_, target_tok_pos, target_pad_mask_pos = prepare_input(model, coords, positive_seq, alphabet, batch_converter, device) # tensor[B, L]
    *_, target_tok_neg, target_pad_mask_neg = prepare_input(model, coords, negative_seq, alphabet, batch_converter, device)

    # masks
    assert torch.equal(target_pad_mask_pos, target_pad_mask_neg),  '[Unexpected] Padding Masks are different'

    padding_mask = target_pad_mask_pos.cpu().detach().numpy() # ndarray[B, L] - True/False
    critical_mask = (target_tok_pos != target_tok_neg).cpu().detach().numpy()

    residue_positions = ~padding_mask # not-padding positions
    mutated_positions = critical_mask * residue_positions # mutated positions
    return target_tok_pos, target_tok_neg, residue_positions, mutated_positions


def score_sequence(log_prob, target_tok_pos, target_tok_neg, sampled_tok, residue_positions, mutated_positions):
    # log_prob = tensor[B, 35, L], sampled_tok = tensor[L], target_tok_pos/neg = [B, L], residue/mutated_position = ndarray(B, L)

    argmax_tok = torch.argmax(log_prob, dim=1) # argmax

    ll_pos, ll_pos_avg = log_likelihood(log_prob, target_tok_pos, residue_positions) # log likelihood of positive sequences
    ll_neg, ll_neg_avg = log_likelihood(log_prob, target_tok_neg, residue_positions) # log likelihood of negative sequences
    ll_argmax, ll_argmax_avg = log_likelihood(log_prob, argmax_tok, residue_positions) # log likelihood of argmax

    rec_pos = sequence_recovery(sampled_tok, target_tok_pos, residue_positions) # sequence recovery of positive sequences
    rec_neg = sequence_recovery(sampled_tok, target_tok_neg, residue_positions) # sequence recovery of negative sequences

    # only at different positions
    _, ll_diff_pos_avg = log_likelihood(log_prob, target_tok_pos, mutated_positions)
    _, ll_diff_neg_avg = log_likelihood(log_prob, target_tok_neg, mutated_positions)
    _, ll_diff_argmax_avg = log_likelihood(log_prob, argmax_tok, mutated_positions)

    rec_diff_pos = sequence_recovery(sampled_tok, target_tok_pos, mutated_positions)
    rec_diff_neg = sequence_recovery(sampled_tok, target_tok_neg, mutated_positions)

    # CSV Format
    metric_dict = {
        "ll_pos": float(ll_pos[0]),
        "ll_pos_avg": float(ll_pos_avg[0]),
        "ll_neg": float(ll_neg[0]),
        "ll_neg_avg": float(ll_neg_avg[0]),
        "ll_argmax": float(ll_argmax[0]),
        "ll_argmax_avg": float(ll_argmax_avg[0]),
        "rec_pos": float(rec_pos[0]),
        "rec_neg": float(rec_neg[0]),
        "ll_diff_pos_avg": float(ll_diff_pos_avg[0]),
        "ll_diff_neg_avg": float(ll_diff_neg_avg[0]),
        "ll_diff_argmax_avg": float(ll_diff_argmax_avg[0]),
        "rec_diff_pos": float(rec_diff_pos[0]),
        "rec_diff_neg": float(rec_diff_neg[0]),
    }

    return metric_dict


def log_likelihood(log_prob, target, valid_mask):
    per_token_logps = log_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)  # (B, L)
    per_token_logps = per_token_logps.cpu().detach().numpy()

    ll_fullseq = np.sum(per_token_logps * valid_mask, axis=1)
    ll_avg = ll_fullseq / np.sum(valid_mask, axis=1)

    return ll_fullseq, ll_avg


def sequence_recovery(sampled_tok, target_tok, valid_mask):
    # sequence recovery with mask
    sampled_tok = sampled_tok.cpu().detach().numpy()
    target_tok = target_tok.cpu().detach().numpy()

    recovery = np.sum((sampled_tok == target_tok) * valid_mask, axis=1) / np.sum(valid_mask, axis=1)
    return recovery

# ------------------
# Main
# ------------------
def main(args):
    log, log_f = make_logger(args.log_file)

    device = args.device if torch.cuda.is_available() else 'cpu'
    model_name = Path(args.model_path).stem if args.model_path else "baseline" # output files are named after the checkpoint

    model, alphabet = load_ckpt_inference(args.model_path, model_name, device=device, log=log)
    model.eval()

    batch_converter = CoordBatchConverter(alphabet)

    assert args.batch_size == 1, "batch_size must be 1 (esm-if model.sample() is implemented for a single backbone)."

    test_ds = ESMIFPreferenceDataset(args.test_pdb_paths, args.test_csv_path) # item = {coords, positive_seq, negative_seq}
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(str(device).startswith("cuda")),
        persistent_workers=(args.num_workers > 0), collate_fn=collate_fn
    )

    run_name = args.run_name or f"{model_name}_T{args.temperature}"
    score_dir, seq_dir, log_prob_dir = ensure_output_paths(f"{args.out_dir}/score_csv", f"{args.out_dir}/sampled_sequences", f"{args.out_dir}/log_probability/{run_name}")

    results = []
    seq_results = []

    for batch in tqdm(test_loader, desc=f"Inference", leave=False):
        seq_id = batch['sequence_ids'][0] # batch_size == 1
        coords = batch['coords'][0] # batch['coords']: [B] * ndarray([L, 3, 3])
        positive_seq = batch['positives'][0] #  batch['positives']: [B]

        all_negative_sequences = batch['all_negatives'] # batch['all_negatives']: 2d list of sequences [B, No_seq]
        negative_seq = [all_negative_sequences[i][0] for i in range(len(all_negative_sequences))][0] # In inference: use the first negative (0-th)

        # check-ups before sampling
        if positive_seq == negative_seq:
            log(f"[Skip][{seq_id}] identical pos/neg. Skipping batch.")
            continue
        if len(positive_seq) != int(coords.shape[0]):
            log(f'[Warning][{seq_id}] coordinates and sequence has different length. Skipping...')
            continue
        if len(positive_seq) != len(negative_seq):
            log(f'[Warning][{seq_id}] Two sequence has different length. Skipping...')
            continue

        # Sampling & Choose Best Sample to Evaluate
        save_path = f"{log_prob_dir}/{seq_id}.pt" if args.save_logprob else None
        sampled_seq_list, sampled_tok_list, log_prob_list, idx = sample_multiple_sequences(model, coords, args.sample_n_seq, args.temperature, device, args.seed, save_path=save_path, positive_seq=positive_seq)
        sampled_seq, sampled_tok, log_prob = sampled_seq_list[idx], sampled_tok_list[idx], log_prob_list[idx] # Best Sample and log probability

        # evaluation metrics
        target_tok_pos, target_tok_neg, residue_positions, mutated_positions = targets_token(model, coords, positive_seq, negative_seq, alphabet, batch_converter, device)
        metrics_dict = score_sequence(log_prob, target_tok_pos, target_tok_neg, sampled_tok, residue_positions, mutated_positions)

        results.append({
        "Sequence_ID": seq_id,
        "GT_sequence": positive_seq,
        "Negative_sequence": negative_seq,
        "Sampled_sequence_best": sampled_seq,
        **metrics_dict,
        })

        sampled_seq_dict = {f"Sampled_sequence_{i}": v for i, v in enumerate(sampled_seq_list)}

        seq_results.append({
        "Sequence_ID": seq_id,
        "GT_sequence": positive_seq,
        "Negative_sequence": negative_seq,
        **sampled_seq_dict,
        })

    df_out = pd.DataFrame(results)
    seq_df_out = pd.DataFrame(seq_results)

    score_csv_path = f"{score_dir}/{run_name}.csv"
    seq_csv_path = f"{seq_dir}/{run_name}.csv"

    df_out.to_csv(score_csv_path, index=False)
    seq_df_out.to_csv(seq_csv_path, index=False)

    log(f"✅ File saved: {score_csv_path}")
    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM-IF Inference")
    # Data
    parser.add_argument("--test_pdb_paths", nargs="+", required=True, help="One or more directories containing test PDB files")
    parser.add_argument("--test_csv_path", type=str, required=True, help="CSV containing test Sequence_ID, GT_sequence, and Neg_seq_{i}")

    # Sampling
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature")
    parser.add_argument("--sample_n_seq", type=int, default=5, help="Number of sequences to sample per structure")

    # System
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=1, choices=[1], help="Batch size, fixed to 1")
    parser.add_argument("--device", type=str, required=True, help="cuda, cpu, or cuda:<index>")

    # I/O
    parser.add_argument("--model_path", type=str, default="", help="Path to checkpoint to load model, empty for the pretrained baseline")
    parser.add_argument("--run_name", type=str, default="", help="Names the output files and the label in src.report (default: <checkpoint>_T<temperature>)")
    parser.add_argument("--out_dir", type=str, default="./inference/outputs", help="Path to save scores, sampled sequences and log-probability matrices (optional)")
    parser.add_argument("--log_file", type=str, default="./logs/evaluate.log", help="Log file path")
    parser.add_argument("--save_logprob", action="store_true", help="Save log-probability matrices for future evaluation")


    args = parser.parse_args()
    main(args)
