"""ESM-IF model-related utilities: forward passes, sampling, and reference logits"""

import os

import numpy as np

import torch
import torch.nn.functional as F

import esm
from esm.inverse_folding.util import CoordBatchConverter 

from src.utils import set_seed

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def prepare_input(model, coords, seqs, alphabet=None, batch_converter=None, device=None):
    if device is None: device = next(model.parameters()).device
    if alphabet is None: alphabet = model.decoder.dictionary
    if batch_converter is None: batch_converter = CoordBatchConverter(alphabet)

    if isinstance(seqs, str):
        seqs = [seqs]
    if isinstance(coords, np.ndarray) and coords.ndim == 3:
        coords = [coords]
    
    batch = [(coord, None, seq) for coord, seq in zip(coords, seqs)]
    
    # coords=tensor[B, L_max+2, 3,3], confidence=tensor([B, L_max+2], tokens=tensor[B, L_max+1], padding_mask=tensor[B,L_max+2])
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
    
    prev_output_tokens = tokens[:, :-1].to(device) # tensor[B, L_max]
    target = tokens[:, 1:].to(device=device, dtype=torch.long)
    target_padding_mask = (target == alphabet.padding_idx)

    return coords, confidence, strs, padding_mask, prev_output_tokens, target, target_padding_mask


def forward_logits(model, coords, seqs, alphabet=None, batch_converter=None, device=None):
    coords, confidence, strs, padding_mask, prev_output_tokens, target, target_padding_mask = prepare_input(model, coords, seqs, alphabet, batch_converter, device)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens) # [B, V, L]
    return logits, target, target_padding_mask


# --------------
# Sample
# --------------
# Modified ESM-IF model.sample to output logit (+ option to not sample, but take argmax)
def sample_with_logit(model, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
    """
    Samples sequences based on multinomial sampling (no beam search).

    Args:
        coords: L x 3 x 3 list representing one backbone
        partial_seq: Optional, partial sequence with mask tokens if part of
            the sequence is known
        temperature: sampling temperature, use low temperature for higher
            sequence recovery and high temperature for higher diversity
        confidence: optional length L list of confidence scores for coordinates
    """
    L = len(coords)
    # Convert to batch format
    batch_converter = CoordBatchConverter(model.decoder.dictionary)
    batch_coords, confidence, _, _, padding_mask = (
        batch_converter([(coords, confidence, None)], device=device)
    )

    logits_list = [] # Added
    
    # Start with prepend token
    mask_idx = model.decoder.dictionary.get_idx('<mask>')
    sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)
    sampled_tokens[0, 0] = model.decoder.dictionary.get_idx('<cath>')
    if partial_seq is not None:
        for i, c in enumerate(partial_seq):
            sampled_tokens[0, i+1] = model.decoder.dictionary.get_idx(c)
        
    # Save incremental states for faster sampling
    incremental_state = dict()
    
    # Run encoder only once
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)
    
    # Make sure all tensors are on the same device if a GPU is present
    if device:
        sampled_tokens = sampled_tokens.to(device)
    
    # Decode one token at a time
    for i in range(1, L+1):
        logits, _ = model.decoder(
            sampled_tokens[:, :i], 
            encoder_out,
            incremental_state=incremental_state,
        )
        logits_list.append(logits.clone()) # Added
        logits = logits[0].transpose(0, 1) # 1, V

        if sampled_tokens[0, i] == mask_idx:  # keep the positions fixed by partial_seq
            if temperature == 0:  # Added
                sampled_tokens[:, i] = torch.argmax(logits, dim=-1)
            else:
                logits /= temperature
                probs = F.softmax(logits, dim=-1)
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
    sampled_seq_idx = sampled_tokens[0, 1:] # idx
    # Convert back to string via lookup
    sampled_seq = ''.join([model.decoder.dictionary.get_tok(a) for a in sampled_seq_idx])
    all_logits = torch.cat(logits_list, dim=-1)
    
    return sampled_seq, sampled_seq_idx, all_logits


# Sample n sequences & Save 
def sample_multiple_sequences(model, coords, n_samples, temperature, device, seed=42, save_path=None, positive_seq=None, partial_seq=None):
    # coords = np.array([L, 3, 3]), positive_seq = L
    set_seed(seed)

    sampled_seq_list = []
    sampled_tok_list = []
    log_prob_list = []
    
    best_idx = -1
    best_recovery = -1.0
    
    for i in range(n_samples):
        with torch.no_grad():
            sampled_seq, sampled_tok, logits = sample_with_logit(
                model, coords, partial_seq=partial_seq, temperature=temperature, device=device
            ) # sampled_tok = tensor[L], logits, log_prob = tensor[B, 35, L]
            
        # Log Probability
        log_prob = F.log_softmax(logits, dim=1) # [B, V, L]
        
        # Save
        sampled_seq_list.append(sampled_seq)
        sampled_tok_list.append(sampled_tok.detach().clone())
        log_prob_list.append(log_prob.detach().clone())
        
        # Find Best (if GT Sequence is given)
        if positive_seq is not None:
            recovery = np.mean([a == b for a, b in zip(positive_seq, sampled_seq)])
            if recovery > best_recovery:
                best_recovery = recovery
                best_idx = i
    
    if save_path:
        save_dict = {
            "sampled_seqs": sampled_seq_list,
            "sampled_tokens": torch.stack(sampled_tok_list, dim=0).cpu(), # tensor[n, L]
            "log_probs": torch.stack(log_prob_list, dim=0).cpu(), # tensor[n, B, 35, L]
            "best_index": best_idx,
        }
        torch.save(save_dict, save_path)
    
    return sampled_seq_list, sampled_tok_list, log_prob_list, best_idx


# ------------------
# Pre-compute Logits
# ------------------
def load_logits_pt(ref_path, sid, nidx, pos_seq=None, neg_seq=None):
    """
    Load reference logits precomputed by the pretrained model, one file per sequence:
        {ref_path}/{sid}.pt = {"sid", "pos": {"seq", "logits", "target"},
                               "negs": [{"seq", "logits", "target"}, ...]}
    where "logits" is [1, V, L] and "negs" follows the order of the Neg_seq_ columns.

    Returns the positive and the nidx-th negative logits, each [1, V, L].
    pos_seq and neg_seq are optional, and only checked against the stored sequences.
    """
    file_path = os.path.join(ref_path, f"{sid}.pt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reference logits not found: {file_path}. See load_logits_pt "
                                f"for the expected payload, or pass --ref_mode model to skip precomputing.")

    data = torch.load(file_path, map_location="cpu")

    # Validate loaded data
    if pos_seq is not None:
        assert data["pos"]["seq"] == pos_seq, f"GT sequence mismatch for {sid}"
    if neg_seq is not None:
        assert data["negs"][nidx]["seq"] == neg_seq, f"Negative sequence mismatch for {sid}, index {nidx}"

    return data["pos"]["logits"], data["negs"][nidx]["logits"]


@torch.no_grad()
def save_logits_pt(output_dir, model, sid, coord, gt_seq, neg_seqs, alphabet, batch_converter, device, log=print):
    """
    For each Sequence_ID in the CSV, compute logits for one GT sequence and all Negative sequences
    Args:
        model: ESM-IF model
        sid: Sequence ID
        coord: np.array of shape [L, 3, 3]
        gt_seq: str, ground-truth sequence
        neg_seqs: List[str], negative sequences
        alphabet: ESM alphabet
        batch_converter: CoordBatchConverter
        device: torch device
    Usage:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model.eval().to(device)
        batch_converter = CoordBatchConverter(alphabet)

        ds = ESMIFPreferenceDataset(pdb_paths, csv_path)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        for batch in loader:
            save_logits_pt(out_dir, model, batch['sequence_ids'][0], batch['coords'][0], batch['positives'][0], batch['all_negatives'][0], alphabet, batch_converter, device)
    """
    p_logits, p_target, _ = forward_logits(model, [coord], [gt_seq], alphabet, batch_converter, device)

    pos_data = {
        "seq": gt_seq,
        "logits": p_logits.cpu().detach(),
        "target": p_target.cpu()
    }

    negs_data_list = []
    for i, n_seq in enumerate(neg_seqs):
        n_logits, n_target, _ = forward_logits(model, [coord], [n_seq], alphabet, batch_converter, device)
        
        negs_data_list.append({
            "seq": n_seq,
            "logits": n_logits.cpu().detach(),
            "target": n_target.cpu()
        })

    payload = {
        "sid": sid,
        "pos": pos_data,
        "negs": negs_data_list,
    }

    save_path = os.path.join(output_dir, f"{sid}.pt")
    torch.save(payload, save_path)
    log(f"[{sid}][Saved] {save_path} with {len(neg_seqs)} negative sequences.")


# ------------------
# Forward Policy / Reference Wrapper
# ------------------
def forward_policy(model, coords, positive_seqs, negative_seqs, alphabet, batch_converter, device):
    logits_w, target_w, pad_mask_w = forward_logits(model, coords, positive_seqs, alphabet, batch_converter, device) # [B, V, L] [B, L]
    logits_l, target_l, pad_mask_l = forward_logits(model, coords, negative_seqs, alphabet, batch_converter, device) # [B, V, L] [B, L]

    assert torch.equal(pad_mask_w, pad_mask_l), "Padding masks should be same for pos/neg for same batch"

    return {
        "logits_w": logits_w, "logits_l": logits_l,
        "target_w": target_w, "target_l": target_l,
        "pad_mask": pad_mask_w,
    }


def forward_reference(ref_model, ref_mode, ref_path, seq_ids, neg_idxs, coords, positive_seqs, negative_seqs, alphabet, batch_converter, device, policy_outputs):
    if ref_mode == "model" and (ref_model is not None):
        results = forward_policy(ref_model, coords, positive_seqs, negative_seqs, alphabet, batch_converter, device)

        assert torch.equal(results['target_w'], policy_outputs['target_w']), "Tokenization should match policy targets"

        return {k: results[k].detach() for k in ("logits_w", "logits_l")}

    elif ref_mode == "precomputed":
        lw_list = []
        ll_list = []

        for sid, nidx in zip(seq_ids, neg_idxs):
            lw, ll = load_logits_pt(ref_path, sid, nidx) # [1, V, L]
            lw_list.append(lw)
            ll_list.append(ll)

        logits_w = torch.cat(lw_list, dim=0).to(device)  # [B, V, L]
        logits_l = torch.cat(ll_list, dim=0).to(device)  # [B, V, L]

        return {"logits_w": logits_w.detach(), "logits_l": logits_l.detach(),}

    else:
        raise ValueError(f"Unknown ref_mode: {ref_mode}. Expected 'model' or 'precomputed'")
