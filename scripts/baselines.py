#!/usr/bin/env python3
"""
baselines_2of4.py

Baselines for strict 2:4 pruning:
- Magnitude (MAG): |W|
- WANDA: |W| * ||X||
- SparseGPT: Weight update minimizing reconstruction error (limited to block-diagonal approximation here).
"""

import argparse
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timesfm

# -------------------------
# Utils
# -------------------------
def find_torch_module(obj) -> nn.Module:
    if isinstance(obj, nn.Module): return obj
    for attr in ("model", "_model", "module", "_module", "torch_model", "_torch_model"):
        m = getattr(obj, attr, None)
        if isinstance(m, nn.Module): return m
    for v in getattr(obj, "__dict__", {}).values():
        if isinstance(v, nn.Module): return v
    raise RuntimeError("Could not locate underlying torch nn.Module.")

def forecast_timesfm_point(tfm_model, X: np.ndarray, horizon: int) -> np.ndarray:
    inputs = [X[i].astype(np.float32) for i in range(X.shape[0])]
    point_forecast, _ = tfm_model.forecast(horizon=horizon, inputs=inputs)
    return np.asarray(point_forecast, dtype=np.float32)

def mse_mae(pred: np.ndarray, tgt: np.ndarray):
    d = pred - tgt
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))

def load_series(csv_path: str, col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if col not in df.columns: raise ValueError(f"Column {col} not found.")
    return df[col].to_numpy(dtype=np.float32)

def make_windows(series: np.ndarray, start: int, end: int, context: int, horizon: int, stride: int):
    xs, ys = [], []
    last = end - (context + horizon)
    for i in range(start, last + 1, stride):
        xs.append(series[i:i+context])
        ys.append(series[i+context:i+context+horizon])
    if not xs: raise ValueError("No windows produced.")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)

def select_linears(torch_mod: nn.Module, include_regex: str, sample_rows: int):
    inc = re.compile(include_regex) if include_regex else None
    out = []
    for name, m in torch_mod.named_modules():
        if isinstance(m, nn.Linear):
            if inc and not inc.match(name): continue
            out.append((name, m))
    return out

# -------------------------
# Gram Collection
# -------------------------
@dataclass
class GramStat:
    Gact: torch.Tensor          # [G,4,4] block-diagonal Gram
    count: int

def collect_grams(tfm_model, targets, X_pool, calib_batch, max_calls, horizon=96):
    stats = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []
    
    # We'll use a simple forward loop. 
    # Warning: TimesFM might micro-batch, but since we just need unweighted Grams, 
    # exact alignment with "window index" doesn't matter, just the distribution of X.
    
    def make_hook(layer_name):
        def pre_hook(_mod, inputs):
            if calls[layer_name] >= max_calls: return
            (x,) = inputs
            
            # Flatten to [N, C]
            if x.dim() == 3: x = x.reshape(-1, x.shape[-1])
            
            # Group into [N, G, 4]
            C = x.shape[-1]
            G = C // 4
            if G * 4 != C: return
            
            xg = x.reshape(-1, G, 4)
            
            # Subsample if too large
            if xg.shape[0] > 2048:
                idx = torch.randint(0, xg.shape[0], (2048,), device=x.device)
                xg = xg.index_select(0, idx)
                
            # Compute G = X^T X per group
            # xg: [N, G, 4] -> einsum -> [G, 4, 4]
            G_batch = torch.einsum("ngc,ngd->gcd", xg, xg).detach().cpu()
            
            if layer_name not in stats:
                stats[layer_name] = GramStat(Gact=G_batch, count=xg.shape[0])
            else:
                stats[layer_name].Gact += G_batch
                stats[layer_name].count += xg.shape[0]
                
            calls[layer_name] += 1
        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    # Drive forward
    n = X_pool.shape[0]
    for i in range(0, n, calib_batch):
        xb = X_pool[i:i+calib_batch]
        _ = forecast_timesfm_point(tfm_model, xb, horizon=horizon)

    for h in hooks: h.remove()
    return stats

# -------------------------
# Pruning
# -------------------------
PAIRS = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.long)

@torch.no_grad()
def prune_2of4_baseline(layer, st: GramStat, mode: str, refit: bool, ridge: float):
    W = layer.weight.data
    O, C = W.shape
    G = C // 4
    if G * 4 != C: return
    device = W.device
    dtype = W.dtype
    
    Wg = W.view(O, G, 4)
    
    if mode == "magnitude":
        # Score = |W|
        scores = Wg.abs()
        # Keep top 2 per group
        # Helper to map 6 pairs
        pair_scores = torch.zeros((O, G, 6), device=device)
        for k in range(6):
            # sum of abs weights for the pair
            # Actually standard magnitude pruning keeps largest weights.
            # 2:4 constraint means for each group of 4, select best 2.
            # So just select indices of 2 largest.
            # Wait, my previous script used pair scoring. Top-2 logic is simpler:
            pass
            
        # Standard top-2 selection
        top2_vals, top2_idx = torch.topk(Wg.abs(), 2, dim=2) # [O, G, 2]
        mask = torch.zeros_like(Wg, dtype=torch.bool)
        mask.scatter_(2, top2_idx, True)

    elif mode == "wanda":
        # Score = |W| * ||X||
        # ||X|| comes from diag(Gact)
        # Gact is [G, 4, 4]
        if st is None: return
        Gact = st.Gact.to(device).float()
        diag = torch.diagonal(Gact, dim1=1, dim2=2) # [G, 4]
        # Normalize by count to get mean X^2 (though relative scale doesn't matter for topk)
        # Actually WANDA uses ||X|| column norms. 
        input_norms = torch.sqrt(diag + 1e-6) # [G, 4]
        
        scores = Wg.abs() * input_norms.unsqueeze(0) # [O, G, 4]
        
        top2_vals, top2_idx = torch.topk(scores, 2, dim=2)
        mask = torch.zeros_like(Wg, dtype=torch.bool)
        mask.scatter_(2, top2_idx, True)

    elif mode == "sparsegpt":
        # SparseGPT: min reconstruction error.
        # Score = W^2 / [H^-1]_ii
        # H = X^T X + damp * I
        if st is None: return
        Gact = st.Gact.to(device).float()
        
        # Damping (1% of mean diag)
        damp = 0.01 * torch.mean(torch.diagonal(Gact, dim1=1, dim2=2))
        eye = torch.eye(4, device=device).unsqueeze(0)
        H = Gact + damp * eye
        
        # Invert H (block-wise)
        Hinv = torch.inverse(H) # [G, 4, 4]
        
        diag_inv = torch.diagonal(Hinv, dim1=1, dim2=2) # [G, 4]
        
        # OBS Score = W^2 / diag(Hinv)
        scores = (Wg ** 2) / (diag_inv.unsqueeze(0) + 1e-10)
        
        # Minimize score (score = increase in error). So we DROP smallest scores. 
        # Wait, if score is "error increase", we want to KEEP indices that would cause large error if dropped.
        # So keep LARGEST scores.
        top2_vals, top2_idx = torch.topk(scores, 2, dim=2)
        mask = torch.zeros_like(Wg, dtype=torch.bool)
        mask.scatter_(2, top2_idx, True)
    
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Apply mask
    if not refit:
        Wnew = Wg * mask
    else:
        # Optimal update: W_new = W - (W H^-1)_{masked} (H^-1_{masked})^-1 ... 
        # Easier way: standard closed form for each block.
        # W_new = argmin || Y - W X ||^2 s.t. Mask
        # Solution: rows of W are independent.
        # For each output o, group g: w = (X_active^T X_active)^-1 X_active^T y
        # Or using precomputed G: w_active = inv(G_active) @ (W_old @ G)_active
        # My previous script refit logic is efficient.
        
        # Let's reuse the block-wise refit logic from snr_2of4...
        # Need "B" matrix = W_old @ G
        Gact = st.Gact.to(device).float()
        eye = torch.eye(4, device=device).unsqueeze(0)
        # Ridge for stability
        H = Gact + ridge * eye # [G, 4, 4]
        
        # B = W G
        B = torch.einsum("ogc,gcd->ogd", Wg, H) # [O, G, 4]
        
        Wnew = torch.zeros_like(Wg)
        
        # We know the mask. For each group, we keep 2 idxs.
        # Let's iterate over the 6 possible pair patterns to batch the inverse
        for k in range(6):
            # Identify which groups use pattern k
            p = PAIRS[k].to(device) # [2]
            # mask matches pattern?
            # efficient check: sum of mask[..., p] == 2?
            # No, strictly match indices used. 
            # Or just iterate output-groups in python?
            # Efficient way:
            # 1. Gather active indices per group. 
            # 2. Extract sub-Gram H_active [G, 2, 2].
            # 3. Inv H_active.
            # 4. W_active = B_active @ inv H_active? No, W_opt = B_projected_to_active @ inv(H_active)
            # W_opt = (target_corr) @ inv(Cov)
            # target_corr = E[y x^T] = W_old E[x x^T] = W_old G = B
            # So W_active = B_active @ inv(H_active). Correct.
            
            # Since mask varies per output channel, this is O * G small solves.
            # Vectorize over (O,G)?
            # mask is [O, G, 4].
            # Indices: top2_idx [O, G, 2]
            
            idx0 = top2_idx[:,:,0] # [O,G]
            idx1 = top2_idx[:,:,1]
            
            # Gather H elements
            # H is [G, 4, 4]. Expand to [O, G, 4, 4] is expensive.
            # But H depends only on G.
            # Selection depends on O.
            
            # Slow loop is acceptable for baselines.
            pass
            
        # Faster approach:
        # Gather B_active: [O, G, 2]
        # Gather H_active: [O, G, 2, 2] (depends on O because mask depends on O)
        
        # Gather H:
        # H expanded: [1, G, 4, 4]
        # We need H[g, idx[o,g,i], idx[o,g,j]]
        
        # Let's utilize the fact that H is shared across O.
        # But indices vary.
        # H_active: [O, G, 2, 2]
        row_idx = torch.arange(G, device=device).view(1, G, 1, 1).expand(O, G, 2, 2)
        idx_r = top2_idx.unsqueeze(3).expand(O, G, 2, 2) # [O,G,2] -> [O,G,2,2]
        idx_c = top2_idx.unsqueeze(2).expand(O, G, 2, 2) # [O,G,2] -> [O,G,2,2]
        
        H_sub = H[row_idx, idx_r, idx_c] # [O, G, 2, 2]
        
        # Invert
        # 2x2 inverse is closed form:
        det = H_sub[...,0,0]*H_sub[...,1,1] - H_sub[...,0,1]*H_sub[...,1,0]
        invHat = torch.empty_like(H_sub)
        inv_det = 1.0 / (det + 1e-10)
        invHat[...,0,0] =  H_sub[...,1,1] * inv_det
        invHat[...,1,1] =  H_sub[...,0,0] * inv_det
        invHat[...,0,1] = -H_sub[...,0,1] * inv_det
        invHat[...,1,0] = -H_sub[...,1,0] * inv_det
        
        # B_active: [O, G, 2]
        # B is [O, G, 4]. Gather cols.
        B_sub = B.gather(2, top2_idx) # [O, G, 2]
        
        # W_active = B_sub @ invHat?
        # Shapes: B_sub [..., 2], invHat [..., 2, 2]
        # matmul ([..., 1, 2], [..., 2, 2]) -> [..., 1, 2]
        W_act = torch.matmul(B_sub.unsqueeze(2), invHat).squeeze(2) # [O, G, 2]
        
        # Scatter back
        Wnew.scatter_(2, top2_idx, W_act)

    layer.weight.data = Wnew.reshape(O, C).to(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--col", default="OT")
    parser.add_argument("--train_end", type=int, default=49152)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride_test", type=int, default=96)
    parser.add_argument("--mode", required=True, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument("--refit", type=int, default=-1, help="-1: auto (0 for Mag/Wanda, 1 for SparseGPT)")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--calib_windows", type=int, default=256)
    parser.add_argument("--calib_select", type=str, default="first", choices=["first", "last"],
                        help="Which end of training data to use for calibration")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    
    # Auto refit
    if args.refit == -1:
        args.refit = 1 if args.mode == "sparsegpt" else 0
        
    print(f"Running {args.mode} (refit={args.refit})...")
    
    # Init Model
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    torch_mod = find_torch_module(tfm)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=args.horizon))
    
    # Load Data
    series = load_series(args.csv, args.col)
    train_end = args.train_end
    X_tr, Y_tr = make_windows(series, 0, train_end, args.context, args.horizon, 1)
    
    # Calib pool (select first or last N windows)
    if args.calib_select == "last":
        X_pool = X_tr[-args.calib_windows:]
    else:
        X_pool = X_tr[:args.calib_windows]
    
    # Collect Grams (if needed)
    targets = select_linears(torch_mod, ".*", 2048)
    stats = None
    if args.mode in ("wanda", "sparsegpt") or args.refit:
        print("Collecting grams...")
        stats = collect_grams(tfm, targets, X_pool, 4, 32, horizon=args.horizon)
        
    # Prune
    print("Pruning...")
    for name, layer in targets:
        st = stats.get(name) if stats else None
        prune_2of4_baseline(layer, st, args.mode, bool(args.refit), args.ridge)
        
    # Eval
    print("Evaluating...")
    X_te, Y_te = make_windows(series, train_end, len(series), args.context, args.horizon, args.stride_test)
    base_preds = []
    # Batch eval
    for i in range(0, len(X_te), 16):
        batch = X_te[i:i+16]
        p = forecast_timesfm_point(tfm, batch, args.horizon)
        base_preds.append(p)
    preds = np.concatenate(base_preds, axis=0)
    mse, mae = mse_mae(preds, Y_te)
    print(f"[{args.mode.upper()}] MSE={mse:.6f} MAE={mae:.6f}")

if __name__ == "__main__":
    main()
