
# (Script content omitted here for brevity in reasoning; inserting full script)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-row sequence logo from logits (ProteinMPNN etc.) with:
  1) Highlight wild-type residue per position
  2) Mark critical-site positions
  3) Compare multiple models/conditions in stacked rows
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import json


try:
    import torch
except ImportError:
    torch = None

ArrayLike = Union[np.ndarray, "torch.Tensor"]

@dataclass
class LogoCfg:
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
    keep_indices: Optional[Sequence[int]] = None
    mode: str = "info"
    batch_reduce: str = "mean"
    color_scheme: str = "chemistry"
    top_k: Optional[int] = None
    wt_marker: str = "^"
    wt_marker_size: float = 18.0
    wt_text: bool = False
    critical_style: str = "vline"
    critical_line_width: float = 1.5
    critical_shade_alpha: float = 0.15
    dpi: int = 300

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _softmax_np(logits, axis=-1):
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def _reduce_batch(P, how):
    return P.mean(axis=0) if how == "mean" else np.median(P, axis=0)

def _topk_filter(P, k):
    out = P.copy()
    for i in range(out.shape[0]):
        idx = np.argsort(out[i])[:-k]
        out[i, idx] = 0.0
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out

def _information_content(P):
    eps = 1e-12
    H = -(P * np.log2(P + eps)).sum(axis=1, keepdims=True)
    R = np.log2(P.shape[1]) - H
    return P * R

def logits_to_df(logits, cfg):
    logits = _to_numpy(logits)
    A = list(cfg.alphabet)
    keep = list(range(len(A)))
    logits = logits[..., keep]
    P = _softmax_np(logits, axis=-1)
    if P.ndim == 3:
        P = _reduce_batch(P, cfg.batch_reduce)
    if cfg.top_k:
        P = _topk_filter(P, cfg.top_k)
    M = _information_content(P) if cfg.mode == "info" else P
    return pd.DataFrame(M, columns=A)

def plot_multimodel_logo(models, wild_type_seq, critical_sites, cfg, out_path, x_ticks_abs=None):
    dfs = [logits_to_df(m["logits"], cfg) for m in models]
    fig, axes = plt.subplots(len(dfs), 1, sharex=True, figsize=(12, 3*len(dfs)))
    idx = 0
    if len(dfs) == 1:
        axes = [axes]
    for ax, df, m in zip(axes, dfs, models):
        logomaker.Logo(df, ax=ax, color_scheme=cfg.color_scheme)
        if idx == 0:
            ax.text(-0.02, -0.05, m["name"], transform=ax.transAxes, ha="left", va="top", fontsize=11)      
        else:
            ax.text(-0.02, -0.1, m["name"], transform=ax.transAxes, ha="left", va="top", fontsize=11)  
        idx = idx + 1    
        y0 = ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for pos in critical_sites:
            ax.axvspan(pos-0.5, pos+0.5, color="mistyrose", alpha=0.85, zorder=-10)
            ax.plot(pos, y0, marker="^", color="red", markersize=6, clip_on=False)
        y = ax.get_ylim()[1]*1.02
        for i in range(len(wild_type_seq)):
            ax.plot(i, y, marker=cfg.wt_marker)
        ax.set_ylabel("Information (bits)" if cfg.mode == "info" else "Probability")
        
        L_win = len(wild_type_seq) # If provided, use absolute residue indices (supports non-contiguous idx)
        if x_ticks_abs is not None:
            breaks = []  # 2) 找断点（idx 不连续的位置）
            for i in range(1, L_win):
                if x_ticks_abs[i] != x_ticks_abs[i - 1] + 1:
                    breaks.append(i)

            ymin, ymax = ax.get_ylim()
            for b in breaks:
                x_break = b - 0.5   # 断点在 b-1 与 b 之间
                ax.text(x_break, ymin, "//", ha="center", va="top", fontsize=12, clip_on=False)
                ax.axvline(x_break, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    for ax in axes:
        ax.set_xlabel("Residue index")

    L_win = len(wild_type_seq) # If provided, use absolute residue indices (supports non-contiguous idx)
    if x_ticks_abs is not None:
        step = 5 # 1) 正常设置数字 ticks（显示绝对位置）
        ticks = list(range(0, L_win, step))
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels([str(x_ticks_abs[t]) for t in ticks])
    else:
        step = 5  # fallback：窗口坐标
        ticks = list(range(0, L_win, step))
        axes[-1].set_xticks(ticks)
    # ---- Add wild-type sequence as a top row above all subplots ----
    top_ax = axes[0]                      # use first axis as reference
    xmin, xmax = top_ax.get_xlim()
    # place letters slightly above top subplot
    y_text = 1.02   # relative to axes height
    for i, aa in enumerate(wild_type_seq):
        top_ax.text(i, y_text, aa, ha="center", va="bottom", fontsize=10, transform=top_ax.get_xaxis_transform())
    top_ax.text(-1.02, y_text, "WT:", ha="right", va="bottom", fontsize=10, transform=top_ax.get_xaxis_transform())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [Line2D([0], [0], marker='^', color='red', linestyle='None', markersize=8, label='Critical site')]
    axes[0].legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()
    return out_path

def slice_window_multi_center(models, wild_type_seq, critical_sites, window=50):
    L_full = models[0]["logits"].shape[-2]
    n = len(critical_sites)
    a = max(5, window // n)  # at least 5 residues per site

    intervals = []
    for c in critical_sites:
        half = a // 2
        s = max(0, c - half)
        e = min(L_full, c + half + 1)
        intervals.append((s, e))

    # merge intervals
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    # flatten indices
    idx = []
    for s, e in merged:
        idx.extend(list(range(s, e)))

    # if still too long, truncate centrally
    if len(idx) > window:
        center = len(idx) // 2
        half = window // 2
        idx = idx[center-half:center+half]

    # slice logits
    sliced_models = []
    for m in models:
        logits = m["logits"]
        if logits.ndim == 2:
            logits_win = logits[idx, :]
        else:
            logits_win = logits[:, idx, :]
        sliced_models.append({"name": m["name"], "logits": logits_win})

    wt_win = "".join([wild_type_seq[i] for i in idx])
    crit_win = [idx.index(c) for c in critical_sites if c in idx]

    return sliced_models, wt_win, crit_win, idx

if __name__ == "__main__":

    # L, K = 250, 20
    # logits1 = np.random.randn(L, K).astype(np.float32)
    # logits2 = np.random.randn(L, K).astype(np.float32）
    # wild_type_seq = ("ACDEFGHIKLMNPQRSTVWY" * 20)[:L]  # ensure length >= L
    # critical_sites = [20, 70, 220]
    # models = [
    #     {"name": "Base model (ESM-IF)", "logits": logits1},
    #     {"name": "FPA (Ours)", "logits": logits2}
    # ]
    logp_file_1 = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/bn0.2/base_entropy.json'
    with open(logp_file_1, 'r') as f:
        loaded_dict_1 = json.load(f)

    # Step 1: Filter items from loaded_dict_1 with Critical_pos length > 10
    filtered_items = [
        item for item in loaded_dict_1
        if item['Critical_pos'] is not None and len(item['Critical_pos']) > 10
    ]

    # Step 2: Get top 100 items by Critical_entropy_per_pos (largest first)
    top_100_items = sorted(filtered_items, key=lambda x: x['Critical_entropy_per_pos'], reverse=True)[:100]

    # Step 3: Load the second file
    logp_file_2 = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/mn0.2_bn0.1_lambda0.2_entropy.json'
    with open(logp_file_2, 'r') as f:
        loaded_dict_2 = json.load(f)

    # Step 4: Create a dictionary for quick lookup of items in loaded_dict_2 by Sequence_ID
    dict_2_by_id = {item['Sequence_ID']: item for item in loaded_dict_2}

    # Step 5: Find the item from top_100_items that has the smallest Critical_entropy_per_pos in loaded_dict_2
    min_item_dict2 = None
    min_entropy = float('inf')

    for item_dict1 in top_100_items:
        seq_id = item_dict1['Sequence_ID']
        
        # Check if this Sequence_ID exists in loaded_dict_2
        if seq_id in dict_2_by_id:
            item_dict2 = dict_2_by_id[seq_id]
            entropy_dict2 = item_dict2['Critical_entropy_per_pos']
            
            if entropy_dict2 < min_entropy:
                min_entropy = entropy_dict2
                min_item_dict2 = item_dict2
                # Also keep the corresponding item from loaded_dict_1
                min_item_dict1 = item_dict1

    # Check if we found a valid item
    if min_item_dict2 is not None:
        # Process the selected items
        logits1 = np.array(min_item_dict1['Log_prob_matrix'])
        critical_sites = min_item_dict1['Critical_pos']
        wild_type_seq = min_item_dict1['GT_sequence'][0]
        
        assert logits1.shape[0] == len(wild_type_seq)
        
        logits2 = np.array(min_item_dict2['Log_prob_matrix'])
        assert logits1.shape == logits2.shape
        assert min_item_dict2['Critical_pos'] == critical_sites
        
        print(f"Selected Sequence_ID: {min_item_dict1['Sequence_ID']}")
        print(f"Loaded_dict_1 entropy: {min_item_dict1['Critical_entropy_per_pos']}")
        print(f"Loaded_dict_2 entropy: {min_item_dict2['Critical_entropy_per_pos']}")
    else:
        print("No matching items found between top 100 and loaded_dict_2")

    models = [
        {"name": "Base model (ProteinMPNN)", "logits": logits1},
        {"name": "FPA (Ours)", "logits": logits2}
    ]

    models_win, wt_win, crit_win, idx = slice_window_multi_center(models, wild_type_seq, critical_sites, window=50)
    cfg = LogoCfg()
    plot_multimodel_logo(models_win, wt_win, crit_win, cfg, "p-mpnn_multimodel_logo.png", x_ticks_abs=idx)
