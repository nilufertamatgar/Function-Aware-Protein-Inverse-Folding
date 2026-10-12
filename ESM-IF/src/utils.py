import os, json
from pathlib import Path
import random

import numpy as np
import torch

import esm
import esm.inverse_folding.util

import biotite
from biotite.structure.io.pdb import PDBFile

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(path, overwrite=False):
    mode = "w" if overwrite else "a"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, mode, buffering=1)
    def log(msg):
        f.write(msg + "\n") # line 35
        f.flush()
    return log, f


def ensure_output_paths(*paths):
    created = []
    for p in paths:
        out_dir = Path(p)
        out_dir.mkdir(parents=True, exist_ok=True)
        created.append(out_dir)
    return created[0] if len(created) == 1 else tuple(created)


# ------------------
# Save / Load helpers
# ------------------
def save_config(save_dir, args_dict, log=print):
    os.makedirs(save_dir, exist_ok=True)
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)
    log(f"[INFO] Saved config to {cfg_path}")


def save_checkpoint(model, optimizer, epoch, loss, save_dir, config=None, log=print):
    os.makedirs(save_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(save_dir))  # keeps stems unique across runs
    path = os.path.join(save_dir, f"{run_name}_epoch_{epoch:03d}.pt")
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,  # always None; kept so every released checkpoint shares one payload format
        "earlystop_state": None,
        "loss": loss,
        "config": config,
    }
    torch.save(payload, path)
    log(f"[INFO] Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, log=print):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    cfg = checkpoint.get("config")
    if cfg is not None:
        log("[INFO] Checkpoint contains config; not overriding CLI, just showing diff opportunity.")
    log(f"[INFO] Loaded checkpoint from {path}, starting at epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint.get("loss", None), cfg


def load_ckpt_inference(ckpt_path, model_name, device="cpu", log=None):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.to(device).eval()

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        log(f"[INFO] Loaded model from {ckpt_path} → model_name={model_name}")

    return model, alphabet

# --------------
# PDB preprocessing
# --------------
def find_pdb_path(paths, pdb_file: str, log=None):
    if isinstance(paths, str): paths = [paths]
    for root in paths:
        cand = os.path.join(root, pdb_file)
        if os.path.exists(cand):
            return cand
    if log: log(f"[Known-Issue] {pdb_file} PDB not found in provided paths! Skipping...")
    raise FileNotFoundError(f"[Error][Known] PDB not found: {pdb_file}")


def load_clean_structure(fpath, chain_id, log=None):
    try:
        structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
        return structure
    except Exception as e:
        pdb = PDBFile.read(fpath)
        structure = pdb.get_structure(model=1)
        all_chains = np.unique([str(c).strip() for c in structure.chain_id])
        
        s_chain = structure[structure.chain_id == chain_id]
        aa_mask = biotite.structure.filter_amino_acids(s_chain)
        n_aa_atoms = int(np.sum(aa_mask))
        
        if n_aa_atoms == 0:
            # No Amino Acid Present in the chain
            if log:
                log(f"[Known-Issue] {Path(fpath).stem}_{chain_id} has no amino-acid residues. Skipping...")
                log(f"Resnames(Before): {np.unique(structure.res_name)}")
                log(f"Resnames(After): {np.unique(s_chain.res_name)}")
            raise RuntimeError(f"[Error][Known] Failed loading structure for {Path(fpath).stem}_{chain_id} (no residue): {str(e)}")
        else:
            if log:
                log(f"[Warning] Failed loading structure for {Path(fpath).stem}_{chain_id}: {str(e)}. Skipping...")
                log(f'Given Chain ID: {chain_id}')
                log(f"Possible Chain ID {all_chains}")
                log(f"Resnames: {np.unique(structure.res_name)}")
            raise RuntimeError(f"[Error] Failed loading structure for {Path(fpath).stem}_{chain_id}: {str(e)}")


def extract_clean_coord(structure, log=None, log_seq_id='the_sequence'):
    # return: coords (numpy.ndarray of shape (L, 3, 3)), native_sequence (str) of len L
    
    # clean structure
    DROP_HET = [atom.res_name for atom in structure if atom.hetero]
    if len(DROP_HET) > 0:
        if log: log(f"[Fixed] Dropped Heteatoms in {log_seq_id}: {DROP_HET}")
    
    #extract coord
    try:
        structure = biotite.structure.array([atom for atom in structure if not atom.hetero]) # Taken from ProteinDPO
        coords, native_sequence = esm.inverse_folding.util.extract_coords_from_structure(structure) #coords, native_sequence
        return coords, native_sequence
    except Exception as e:
        # filter again based on canonical amino acids
        aa_mask = biotite.structure.filter_canonical_amino_acids(structure) # filter GHP, 3MY etc.
        DROP_RES = np.unique(structure.res_name[~aa_mask]).tolist()
        structure = structure[aa_mask]
        
        if "PYL" in np.unique(structure.res_name):   
            structure = biotite.structure.array([atom for atom in structure if not atom.res_name == 'PYL'])
            DROP_RES.append('PYL(Additional Filter)')
        
        # unique residues in 
        res_keys = np.unique(np.stack([structure.chain_id, structure.res_id], axis=1), axis=0)

        if len(res_keys) <= 1 or len(structure) == 0:
            if log: log(f"[Known-Issue] Too few residues after filtering in {log_seq_id}: residues-{res_keys} (n={len(res_keys)}). (Filtered residues: {DROP_RES}) Skipping...")
            raise RuntimeError(f"[Error][Known] extract_coords failed for {log_seq_id} (few residue): {str(e)}") 
        else:
            try:
                coords, native_sequence = esm.inverse_folding.util.extract_coords_from_structure(structure)
                if log: log(f"[Fixed] Additionally dropped in {log_seq_id}: {DROP_RES}")
                return coords, native_sequence
            except Exception as e2:
                if log:
                    log(f"[Warning] Failed extracting coords for {log_seq_id}: {str(e2)}") 
                    log(f"{structure}")
                    log(f"resnames: {np.unique(structure.res_name)}")  
                raise RuntimeError(f"[Error] extract_coords failed for {log_seq_id}: {str(e2)}") 


def get_coords_and_sequence(fpath, chain_id, log=None):
    seq_id = f"{Path(fpath).stem}_{chain_id}"
    
    structure = load_clean_structure(fpath, chain_id, log)
    coords, native_sequence = extract_clean_coord(structure, log, seq_id)
    
    return coords, native_sequence


def find_all_chains(fpath):
    pdb = PDBFile.read(fpath)
    structure = pdb.get_structure(model=1)
    all_chains = np.unique([str(c).strip() for c in structure.chain_id])
    return all_chains
