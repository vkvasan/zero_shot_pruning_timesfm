#!/usr/bin/env python3
"""
chronos_benchmark_universal.py

STRATEGY: "Hard-Gated" Spectral Pruning
1. Calculate Spectral Flatness (SFM) per channel.
2. THE SWITCH:
   - If SFM > 0.25 (Spiky data like ETTm2): Use Magnitude Pruning (Score = |W|).
   - If SFM < 0.25 (Smooth data like Weather): Use Spectral Pruning (Score = |W| * SNR).
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import random
import os

try:
    from chronos import ChronosPipeline
except ImportError:
    raise ImportError("Please install chronos-ts: pip install chronos-ts")

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

class UniversalCollector:
    def __init__(self, model, layers, device):
        self.stats = {}
        self.hooks = []
        self.device = device
        for name, module in layers:
            self.stats[name] = {
                "G_sum": None, "count": 0,
                "trend_energy": 0, "season_energy": 0, "noise_energy": 0,
                "sfm_sum": 0 
            }
            h = module.register_forward_pre_hook(self.make_hook(name))
            self.hooks.append(h)
            
    def make_hook(self, name):
        def hook(module, args):
            if not args: return
            x = args[0].detach() 
            
            # --- 1. WANDA STATS ---
            if x.dim() == 3: x_flat = x.reshape(-1, x.shape[-1])
            else: x_flat = x
            
            if self.stats[name]["G_sum"] is None:
                dim = x_flat.shape[1] // 4
                self.stats[name]["G_sum"] = torch.zeros((dim, 4, 4), device=self.device, dtype=torch.float32)
            
            x_blk = x_flat.view(-1, x_flat.shape[1]//4, 4).float()
            G_batch = torch.einsum("bgi,bgj->gij", x_blk, x_blk)
            self.stats[name]["G_sum"] += G_batch
            
            # --- 2. SPECTRAL STATS ---
            if x.dim() == 3: 
                x_float = x.float() 
                x_fft = torch.fft.rfft(x_float, dim=1)
                energy = x_fft.abs().pow(2) 
                
                # SFM Calculation (Geometric Mean / Arithmetic Mean)
                psd = energy + 1e-12
                geo_mean = torch.exp(torch.mean(torch.log(psd), dim=1))
                ari_mean = torch.mean(psd, dim=1)
                sfm = geo_mean / (ari_mean + 1e-12)
                self.stats[name]["sfm_sum"] += sfm.sum(dim=0)
                
                # Component Extraction
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

def apply_magnitude(model):
    print("  > Pruning: Magnitude (2:4)")
    for name, layer in get_prunable_layers(model):
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        _, idx = torch.topk(W_view.abs(), 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

def apply_wanda(model, stats):
    print("  > Pruning: Wanda (2:4)")
    for name, layer in get_prunable_layers(model):
        if name not in stats: continue
        s = stats[name]
        if s["count"] == 0: continue
        G = s["G_sum"] / s["count"]
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        diag_G = torch.diagonal(G, dim1=1, dim2=2)
        inp_norm = torch.sqrt(diag_G + 1e-6).unsqueeze(0).to(W.dtype)
        score = W_view.abs() * inp_norm
        _, idx = torch.topk(score, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

def apply_sparsegpt(model, stats, device):
    print("  > Pruning: SparseGPT (2:4)")
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
        scores = W_view.pow(2) / (inv_diag + 1e-9)
        _, idx = torch.topk(scores, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

def apply_spectral_hard_gated(model, stats, alpha=1.0, beta=1.0, gamma=1.0):
    """
    HARD GATE STRATEGY:
    - If SFM > 0.25 (Spiky): Force Quality=1.0 (Magnitude Pruning)
    - If SFM < 0.25 (Smooth): Use Spectral Quality (Denoising)
    """
    print("  > Pruning: Spectral Hard-Gated (Switching via Flatness)")
    for name, layer in get_prunable_layers(model):
        if name not in stats: continue
        s = stats[name]
        cnt = s["count"]
        if cnt == 0: continue
        
        # 1. Calculate SFM
        sfm_avg = s["sfm_sum"] / cnt 
        
        # 2. Calculate Spectral Quality
        E_trend = s["trend_energy"] / cnt
        E_season = s["season_energy"] / cnt
        E_noise = s["noise_energy"] / cnt
        
        signal = (alpha * E_trend) + (beta * E_season)
        spectral_calc = signal / (E_noise.pow(gamma) + 1e-9)
        spectral_calc = spectral_calc / (spectral_calc.median() + 1e-9)
        spectral_calc = torch.clamp(spectral_calc, 0.1, 10.0)
        
        # 3. THE HARD SWITCH
        # If SFM > 0.25, use 1.0. Else use spectral_calc.
        is_spiky = sfm_avg > 0.25
        
        # We need to construct the Quality Vector
        # Start with Spectral values
        quality = spectral_calc
        # Overwrite the spiky channels with 1.0
        quality[is_spiky] = 1.0
        
        # 4. Apply
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        
        Q_exp = quality.unsqueeze(0).expand(O, C).to(W.dtype)
        Q_view = Q_exp.view(O, -1, 4)
        
        # Score = |W| * sqrt(Quality)
        # For Spiky channels, sqrt(1.0) = 1.0, so Score = |W| (Magnitude Pruning)
        score = W_view.abs() * torch.sqrt(Q_view)
        
        _, idx = torch.topk(score, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

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
    print(f"=== CHRONOS PRUNING SHOWDOWN (Hard-Gated) ===")
    
    try:
        df = pd.read_csv(args.csv_path).select_dtypes(include=[np.number]).dropna(axis=1).fillna(0)
    except:
        print("Error loading CSV.")
        return

    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-large", device_map=args.device, torch_dtype=torch.bfloat16)
    orig_state = {k: v.cpu().clone() for k, v in pipeline.model.state_dict().items()}

    print("\n[1] Calibration & Spectral Analysis...")
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
    except Exception as e:
        print(f"Calibration Error: {e}")
        return
    collector.close()

    results = {}
    print("\n--- 1. DENSE ---")
    results["Dense"] = evaluate(pipeline, df, args.device)

    print("\n--- 2. MAGNITUDE ---")
    pipeline.model.load_state_dict(orig_state)
    apply_magnitude(pipeline.model)
    results["Magnitude"] = evaluate(pipeline, df, args.device)

    print("\n--- 3. WANDA ---")
    pipeline.model.load_state_dict(orig_state)
    apply_wanda(pipeline.model, collector.stats)
    results["Wanda"] = evaluate(pipeline, df, args.device)

    print("\n--- 4. SPARSEGPT ---")
    pipeline.model.load_state_dict(orig_state)
    apply_sparsegpt(pipeline.model, collector.stats, args.device)
    results["SparseGPT"] = evaluate(pipeline, df, args.device)

    print("\n--- 5. SPECTRAL GATED (Hard Switch) ---")
    pipeline.model.load_state_dict(orig_state)
    apply_spectral_hard_gated(pipeline.model, collector.stats)
    results["Spectral_Gated"] = evaluate(pipeline, df, args.device)

    print("\n" + "="*40)
    print(f"{'Method':<15} | {'MSE (Lower is Better)':<20}")
    print("-" * 40)
    for m, score in results.items():
        print(f"{m:<15} | {score:<20.4f}")
    print("="*40)

if __name__ == "__main__":
    main()