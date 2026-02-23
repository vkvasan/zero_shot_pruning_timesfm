#!/usr/bin/env python3
"""
chronos_universal_v2.py

COMPETITORS:
1. MAG (Magnitude)
2. WANDA (Weights + Activations)
3. SPARSEGPT (Hessian Inverse)
4. UNIVERSAL (Your Adaptive Spectral Method)

Run this on ALL datasets to get the final table.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import copy

try:
    from chronos import ChronosPipeline
except ImportError:
    raise ImportError("Please install chronos-ts: pip install chronos-ts")

# --- UTILS ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_prunable_layers(model):
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.in_features % 4 == 0:
            layers.append((name, mod))
    return layers

# --- COLLECTOR ---
class UniversalCollector:
    def __init__(self, model, layers, device):
        self.stats = {}
        self.hooks = []
        self.device = device
        for name, module in layers:
            self.stats[name] = {
                # Shared Stats for Wanda/SparseGPT
                "G_sum": None, 
                # Stats for Universal (Spectral)
                "trend_energy": 0, "season_energy": 0, "noise_energy": 0,
                "sfm_sum": 0, "count": 0
            }
            h = module.register_forward_pre_hook(self.make_hook(name))
            self.hooks.append(h)
            
    def make_hook(self, name):
        def hook(module, args):
            if not args: return
            x = args[0].detach()
            
            # 1. Hessian / Activation Collection (For Wanda & SparseGPT)
            if x.dim() == 3: x_flat = x.reshape(-1, x.shape[-1])
            else: x_flat = x
            
            # Initialize Hessian accumulator if needed
            if self.stats[name]["G_sum"] is None:
                dim = x_flat.shape[1] // 4
                self.stats[name]["G_sum"] = torch.zeros((dim, 4, 4), device=self.device, dtype=torch.float32)
            
            # Compute G = X^T * X (Block-wise approximation for speed)
            x_blk = x_flat.view(-1, x_flat.shape[1]//4, 4).float()
            G_batch = torch.einsum("bgi,bgj->gij", x_blk, x_blk)
            self.stats[name]["G_sum"] += G_batch

            # 2. Spectral Analysis (For Universal Method)
            if x.dim() == 3: 
                x_float = x.float() 
                x_fft = torch.fft.rfft(x_float, dim=1)
                energy = x_fft.abs().pow(2) 
                
                # Spectral Flatness Measure (SFM)
                psd = energy + 1e-12
                geo_mean = torch.exp(torch.mean(torch.log(psd), dim=1))
                ari_mean = torch.mean(psd, dim=1)
                sfm = geo_mean / (ari_mean + 1e-12)
                self.stats[name]["sfm_sum"] += sfm.sum(dim=0)
                
                # Frequency Band Power
                freq_len = energy.shape[1]
                idx_trend = int(freq_len * 0.05)
                idx_noise = int(freq_len * 0.70)
                
                trend = energy[:, :idx_trend, :].sum(dim=(0, 1))
                noise = energy[:, idx_noise:, :].sum(dim=(0, 1))
                mid_band = energy[:, idx_trend:idx_noise, :]
                
                if mid_band.shape[1] > 0:
                    top_k, _ = torch.topk(mid_band, k=min(3, mid_band.shape[1]), dim=1)
                    season = top_k.sum(dim=(0, 1))
                else:
                    season = torch.zeros_like(trend)
                    
                self.stats[name]["trend_energy"] += trend
                self.stats[name]["season_energy"] += season
                self.stats[name]["noise_energy"] += noise
            
            self.stats[name]["count"] += x.shape[0]
        return hook

    def close(self):
        for h in self.hooks: h.remove()

# --- PRUNERS ---

# 1. MAG (Magnitude)
def apply_mag(model):
    print("  > Pruning: MAG (Magnitude)")
    for name, layer in get_prunable_layers(model):
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        _, idx = torch.topk(W_view.abs(), 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

# 2. WANDA (Weights and Activation)
def apply_wanda(model, stats):
    print("  > Pruning: WANDA")
    for name, layer in get_prunable_layers(model):
        if name not in stats: continue
        s = stats[name]
        if s["count"] == 0: continue
        
        # Calculate Input Norm from the Hessian Diagonal (Efficient Reuse)
        G = s["G_sum"] / s["count"]
        # diag(X^T X) is exactly sum(x^2), which is ||x||^2
        # Wanda uses ||x|| (L2 norm)
        input_norm_sq = torch.diagonal(G, dim1=1, dim2=2)
        input_norm = torch.sqrt(input_norm_sq + 1e-6)
        
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        
        # Wanda Score = |W| * ||X||
        inp_norm_exp = input_norm.unsqueeze(0).to(W.dtype)
        score = W_view.abs() * inp_norm_exp
        
        _, idx = torch.topk(score, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

# 3. SPARSEGPT
def apply_sparsegpt(model, stats, device):
    print("  > Pruning: SPARSEGPT")
    for name, layer in get_prunable_layers(model):
        if name not in stats: continue
        s = stats[name]
        if s["count"] == 0: continue
        
        G = s["G_sum"] / s["count"]
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        
        diag_mean = torch.mean(torch.diagonal(G, dim1=1, dim2=2))
        damp = 0.01 * diag_mean
        eye = torch.eye(4, device=device).unsqueeze(0).expand(G.shape[0], 4, 4)
        G_damped = G + damp * eye
        
        try: H_inv = torch.linalg.inv(G_damped)
        except: H_inv = torch.linalg.pinv(G_damped)
        
        inv_diag = torch.diagonal(H_inv, dim1=1, dim2=2).unsqueeze(0).to(W.dtype)
        
        # SparseGPT Score = W^2 / [H^-1]_ii
        scores = W_view.pow(2) / (inv_diag + 1e-9)
        
        _, idx = torch.topk(scores, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

# 4. UNIVERSAL (Your Adaptive Method)
def apply_universal(model, stats):
    print(f"  > Pruning: UNIVERSAL (Adaptive Spectral)")
    spectral_dominance = []
    
    for name, layer in get_prunable_layers(model):
        if name not in stats: continue
        s = stats[name]
        cnt = s["count"]
        if cnt == 0: continue
        
        # A. Chaos Check
        sfm_avg = s["sfm_sum"] / cnt 
        # Confidence: High if SFM < 0.2 (Periodic), Low if SFM > 0.6 (Chaos)
        confidence = 1.0 - torch.clamp((sfm_avg - 0.2) / 0.4, 0.0, 1.0)
        spectral_dominance.append(confidence.mean().item())

        # B. Spectral Quality
        E_trend = s["trend_energy"] / cnt
        E_season = s["season_energy"] / cnt
        E_noise = s["noise_energy"] / cnt
        
        signal = E_trend + E_season
        raw_quality = signal / (E_noise + 1e-9)
        quality_score = raw_quality / (raw_quality.median() + 1e-9)
        quality_score = torch.clamp(quality_score, 0.1, 10.0)
        
        # C. Soft Blend: (Conf * Spectral) + ((1-Conf) * Magnitude)
        Q_expanded = quality_score.unsqueeze(0)
        Conf_expanded = confidence.unsqueeze(0)
        Final_Q = (Conf_expanded * Q_expanded) + ((1.0 - Conf_expanded) * 1.0)
        
        # D. Apply
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        Q_final_exp = Final_Q.expand(O, C).to(W.dtype).view(O, -1, 4)
        
        score = W_view.abs() * torch.sqrt(Q_final_exp)
        _, idx = torch.topk(score, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)
        
    avg_conf = sum(spectral_dominance) / (len(spectral_dominance) + 1e-9)
    print(f"    [Info] Data Periodicity Confidence: {avg_conf*100:.1f}%")

# --- EVALUATION ---
def evaluate(pipeline, df, device):
    cols = df.columns[:32]
    total_mse = 0
    count = 0
    for col in cols:
        series = torch.tensor(df[col].values, dtype=torch.float32)
        if len(series) < 600: continue
        ctx = series[-600:-96]
        act = series[-96:].to(device)
        with torch.no_grad():
            f = pipeline.predict([ctx], 96, num_samples=1)
            pred = torch.tensor(np.stack(f)).squeeze(1).to(device)
            mse = (pred - act).pow(2).mean().item()
            total_mse += mse
            count += 1
    return total_mse / max(count, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    
    # 1. Load Data
    try:
        df = pd.read_csv(args.csv_path).select_dtypes(include=[np.number]).dropna(axis=1).fillna(0)
        print(f"=== BENCHMARK: {os.path.basename(args.csv_path)} ===")
    except: return

    # 2. Load Model & Calibrate
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-large", device_map=args.device, torch_dtype=torch.bfloat16)
    orig_state = {k: v.cpu().clone() for k, v in pipeline.model.state_dict().items()}

    print("\n[1] Calibration (Collecting Hessians & Spectra)...")
    layers = get_prunable_layers(pipeline.model)
    collector = UniversalCollector(pipeline.model, layers, args.device)
    pipeline.model.eval()
    rng = np.random.default_rng(42)
    cols = list(df.columns)
    try:
        for _ in tqdm(range(64)):
            c = rng.choice(cols)
            series = torch.tensor(df[c].values, dtype=torch.float32)
            if len(series) < 600: continue
            start = rng.integers(0, len(series)-600)
            ctx = series[start:start+512]
            with torch.no_grad(): pipeline.predict([ctx], 96)
    except: pass
    collector.close()

    results = {}

    # 3. Run All Methods
    print("\n[2] Evaluating Competitors...")

    # A. MAG
    pipeline.model.load_state_dict(orig_state)
    apply_mag(pipeline.model)
    results["MAG"] = evaluate(pipeline, df, args.device)
    print(f"MAG MSE:       {results['MAG']:.4f}")

    # B. WANDA
    pipeline.model.load_state_dict(orig_state)
    apply_wanda(pipeline.model, collector.stats)
    results["WANDA"] = evaluate(pipeline, df, args.device)
    print(f"WANDA MSE:     {results['WANDA']:.4f}")

    # C. SPARSEGPT
    pipeline.model.load_state_dict(orig_state)
    apply_sparsegpt(pipeline.model, collector.stats, args.device)
    results["SPARSEGPT"] = evaluate(pipeline, df, args.device)
    print(f"SPARSEGPT MSE: {results['SPARSEGPT']:.4f}")

    # D. UNIVERSAL (Yours)
    pipeline.model.load_state_dict(orig_state)
    apply_universal(pipeline.model, collector.stats)
    results["UNIVERSAL"] = evaluate(pipeline, df, args.device)
    print(f"UNIVERSAL MSE: {results['UNIVERSAL']:.4f}")

    # 4. Final Table
    print("\n" + "="*60)
    print(f"{'Method':<20} | {'MSE (Lower=Better)':<20} | {'Status'}")
    print("-" * 60)
    best_score = min(results.values())
    for m, score in results.items():
        status = "**WINNER**" if score == best_score else ""
        print(f"{m:<20} | {score:<20.4f} | {status}")
    print("="*60)

if __name__ == "__main__":
    main()