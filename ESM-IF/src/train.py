import argparse
import copy
import time

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import esm
from esm.inverse_folding.util import CoordBatchConverter

from src.utils import set_seed, make_logger, save_checkpoint, load_checkpoint, save_config
from src.objectives import LOSS_NAMES, compute_mask, get_batch_logps, preference_loss
from src.dataset import ESMIFPreferenceDataset, collate_fn
from src.model import disable_dropout, forward_policy, forward_reference


def get_batch_metrics(policy_model, reference_model, batch, neg_idxs, loss_config, ref_mode, ref_path, alphabet, batch_converter, device):
    # --------- inputs --------
    seq_ids = batch['sequence_ids'] # [B]
    coords = batch['coords'] # [B] * np.array([L, 3, 3])
    positive_seqs = batch['positives'] # [B]
    negative_seqs = [batch['all_negatives'][i][neg_idxs[i]] for i in range(len(batch['all_negatives']))] # [B][max 20]
    
    avg_loss = loss_config['avg']

    # --------- compute logits --------
    # policy forward (logits_w, l: [B, V, L], target_w, l, pad_mask: [B, L]) 
    pol = forward_policy(policy_model, coords, positive_seqs, negative_seqs, alphabet, batch_converter, device)

    # reference forward (logits_w, l: [B, V, L])
    with torch.no_grad():
        ref = forward_reference(reference_model, ref_mode, ref_path, seq_ids, neg_idxs, coords, positive_seqs, negative_seqs, alphabet, batch_converter, device, pol)
    
    # transpose
    for k in ("logits_w", "logits_l"): 
        pol[k], ref[k] = pol[k].transpose(1, 2), ref[k].transpose(1, 2)  # logit: [B, V, L] -> [B, L, V]

    logit_target_dict = {
        "w": (pol["logits_w"], pol["target_w"]),
        "l": (pol["logits_l"], pol["target_l"]),
        "ref_w": (ref["logits_w"], pol["target_w"]),
        "ref_l": (ref["logits_l"], pol["target_l"]),
    }

    # --------- per_token_logps (+ masking) --------
    # masking
    loss_mask, residue_mask = compute_mask(pol['pad_mask'], pol['target_w'], pol['target_l'], loss_config['masked']) # [B, L]

    # batch logps with masking (policy and referenece)
    logps = {}
    for key, (logits, target) in logit_target_dict.items():
        logps[key] = get_batch_logps(logits, target, avg_loss, mask=loss_mask) # lops[key]: [B]

    # --------- calculate loss --------
    # preference loss 
    loss, metrics = preference_loss(
        logps["w"], logps["l"], logps["ref_w"], logps["ref_l"],
        loss_name= loss_config['name'],
        beta = loss_config['beta'],
        lambda_ce = loss_config['lambda_ce'],
        logits_w = pol["logits_w"], target_w = pol["target_w"], residue_mask = residue_mask) # CE input

    return loss, metrics


# -------------------------
# Main
# -------------------------
def main(args):
    run_config = vars(args).copy()
    save_config(args.save_dir, run_config)

    set_seed(args.seed)
    log, log_f = make_logger(args.log_file)
    device = args.device if torch.cuda.is_available() else 'cpu'
    # FPA restricts the preference term to mutated residues by definition (Eq. 8).
    fpa = args.loss_name == "fpa"
    loss_config = {"name": args.loss_name, "beta": args.beta, "lambda_ce": args.lambda_ce,
                   "masked": args.mask or fpa, "avg": args.loss_avg}

    # ---------- Load Data ---------
    train_ds = ESMIFPreferenceDataset(args.train_pdb_paths, args.train_csv_path) # item = {coords, positive_seq, negative_seq}
    val_ds = ESMIFPreferenceDataset(args.val_pdb_paths, args.val_csv_path)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0), collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0), collate_fn=collate_fn
    )

    # ---------- Load Model ---------
    # Policy
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.to(device)
    
    # Reference
    ref_model = None
    if args.ref_mode == "model":
        ref_model = copy.deepcopy(model)
        ref_model.eval().to(device)
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # disable dropout (experiment)
    if args.disable_dropout:
        disable_dropout(model)
        if ref_model is not None:
            disable_dropout(ref_model)

    batch_converter = CoordBatchConverter(alphabet)

    # ---------- Optimizer ---------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=args.adam_betas,
                                  eps=args.adam_epsilon, weight_decay=args.adam_weight_decay)

    # ---------- Load Checkpoint ---------
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint_epoch, best_val_loss, _ = load_checkpoint(args.resume, model, optimizer, log=log)
        start_epoch = checkpoint_epoch + 1
    
    # ----- Training Loop -----
    for epoch in range(start_epoch, args.epochs):
        train_loss = 0.0
        val_loss = 0.0

        # --------- train --------
        model.train()
        train_start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            neg_idxs = [epoch % len(neg_candidates) for neg_candidates in batch['all_negatives']]

            optimizer.zero_grad()

            loss, metrics = get_batch_metrics(model, ref_model, batch, neg_idxs, loss_config, args.ref_mode, args.train_ref_path, alphabet, batch_converter, device)

            loss.backward()
            optimizer.step()

            # logging
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start))

        # --------- validation --------
        model.eval()
        val_start = time.time()

        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            neg_idxs = [epoch % len(neg_candidates) for neg_candidates in batch['all_negatives']]

            with torch.no_grad():
                loss, metrics = get_batch_metrics(model, ref_model, batch, neg_idxs, loss_config, args.ref_mode, args.val_ref_path, alphabet, batch_converter, device)

            val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        val_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - val_start))
        current_lr = optimizer.param_groups[0]['lr']

        log(f"Epoch {epoch:03d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | train_time={train_elapsed} | val_time={val_elapsed} | learning rate={current_lr}")

        # --- Save Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_best:
                save_checkpoint(model, optimizer, epoch, val_loss, args.save_dir, config=run_config, log=log)
                log(f"[INFO] 🏆 New best val loss={best_val_loss:.6f} at epoch {epoch}")
        elif args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, args.save_dir, config=run_config, log=log)
        log("")
    log_f.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune ESM-IF with Preference Dataset")

    # Core training args
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable_dropout", action="store_true", help="Disable dropout during training")
    parser.add_argument("--lr", type=float, default=1.0e-07, help="Learning rate")

    # Loss
    parser.add_argument("--loss_name", type=str, choices=list(LOSS_NAMES), default="fpa", help="Preference objective")
    parser.add_argument("--mask", action="store_true", default=False, help="Restrict the preference term to mutated residues")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta of the preference term")
    parser.add_argument("--lambda_ce", type=float, default=1.0, help="Weight of the CE term in FPA")
    parser.add_argument("--loss_avg", action="store_true", default=False, help="Average log-probabilities instead of summing")

    # Optimizer (AdamW)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.98), help="Betas for the AdamW (default: 0.9 0.98)")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for the AdamW (default: 1e-8)")
    parser.add_argument("--adam_weight_decay", type=float, default=0.1, help="AdamW Weight decay")

    # Reference options
    parser.add_argument("--ref_mode", type=str, choices=["model", "precomputed"], default="model", help="Reference model: a frozen copy of the model, or precomputed logits with model/save_logits_pt (equivalent)")
    parser.add_argument("--train_ref_path", type=str, default="", help="Path to precomputed train reference logits (.pt)")
    parser.add_argument("--val_ref_path", type=str, default="", help="Path to precomputed validation reference logits (.pt)")

    # Data paths
    parser.add_argument("--train_pdb_paths", nargs="+", required=True, help="One or more directories containing train PDB files")
    parser.add_argument("--val_pdb_paths", nargs="+", required=True, help="One or more directories containing validation PDB files")
    parser.add_argument("--train_csv_path", type=str, required=True, help="CSV containing train Sequence_ID, GT_sequence, and Neg_seq_{i}")
    parser.add_argument("--val_csv_path", type=str, required=True, help="CSV containing validation Sequence_ID, GT_sequence, and Neg_seq_{i}")

    # System / DataLoader
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, required=True, help="cuda, cpu, or cuda:<index>")

    # ---- Model I/O ----
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--save_best", action="store_true", default=False, help="Save best checkpoint by validation loss")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # ---- Logging ----
    parser.add_argument("--log_file", type=str, default="./logs/train.log", help="Log file path")

    args = parser.parse_args()

    main(args)