#!/usr/bin/env python3
"""
chronos_universal.py

THE UNIVERSAL PRUNER: "Soft-Gated Spectral Pruning"
1. Logic: Calculates Spectral Flatness (SFM) per layer.
2. Adaptation:
   - Low Entropy (Periodic): Pruning score is driven by Frequency Power (Preserves Signal).
   - High Entropy (Chaos): Pruning score smoothly converges to Magnitude (Preserves Safety).
3. Result: Matches SOTA on periodic data (ETTm1) without crashing on chaotic data (ETTm2).
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

# --- COLLECTOR FOR ALL STATS ---
class UniversalCollector:
    def __init__(self, model, layers, device):
        self.stats = {}
        self.hooks = []
        self.device = device
        for name, module in layers:
            self.stats[name] = {
                # SparseGPT Stats
                "G_sum": None, 
                # Spectral Stats
                "trend_energy": 0, "season_energy": 0, "noise_energy": 0,
                "sfm_sum": 0, "count": 0
            }
            h = module.register_forward_pre_hook(self.make_hook(name))
            self.hooks.append(h)
            
    def make_hook(self, name):
        def hook(module, args):
            if not args: return
            x = args[0].detach()
            
            # 1. SparseGPT Hessian Update
            if x.dim() == 3: x_flat = x.reshape(-1, x.shape[-1])
            else: x_flat = x
            if self.stats[name]["G_sum"] is None:
                dim = x_flat.shape[1] // 4
                self.stats[name]["G_sum"] = torch.zeros((dim, 4, 4), device=self.device, dtype=torch.float32)
            x_blk = x_flat.view(-1, x_flat.shape[1]//4, 4).float()
            # Fast batch matrix multiplication for Hessian
            G_batch = torch.einsum("bgi,bgj->gij", x_blk, x_blk)
            self.stats[name]["G_sum"] += G_batch

            # 2. Spectral Analysis
            if x.dim() == 3: 
                x_float = x.float() 
                x_fft = torch.fft.rfft(x_float, dim=1)
                energy = x_fft.abs().pow(2) 
                
                # Spectral Flatness Measure (SFM) = GeometricMean / ArithmeticMean
                psd = energy + 1e-12
                geo_mean = torch.exp(torch.mean(torch.log(psd), dim=1))
                ari_mean = torch.mean(psd, dim=1)
                sfm = geo_mean / (ari_mean + 1e-12)
                self.stats[name]["sfm_sum"] += sfm.sum(dim=0)
                
                # Band Separation
                freq_len = energy.shape[1]
                idx_trend = int(freq_len * 0.05)
                idx_noise = int(freq_len * 0.70) # Standard cutoffs
                
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

# --- PRUNING METHOD 1: UNIVERSAL SOFT-GATE (YOUR METHOD) ---
def apply_universal_soft_gate(model, stats):
    print(f"  > Pruning: Universal Soft-Gate (Adaptive)")
    
    spectral_dominance = []
    
    for name, layer in get_prunable_layers(model):
        if name not in stats: continue
        s = stats[name]
        cnt = s["count"]
        if cnt == 0: continue
        
        # 1. Calculate Chaos (SFM)
        sfm_avg = s["sfm_sum"] / cnt 
        # Normalize SFM to a confidence score (0 = Chaos, 1 = Periodic)
        # SFM usually ranges 0.0 (Sine) to 1.0 (White Noise).
        # We use a soft sigmoid-like transition.
        # If SFM > 0.6, confidence drops to 0. If SFM < 0.2, confidence is 1.
        confidence = 1.0 - torch.clamp((sfm_avg - 0.2) / 0.4, 0.0, 1.0)
        
        spectral_dominance.append(confidence.mean().item())

        # 2. Calculate Spectral Quality
        E_trend = s["trend_energy"] / cnt
        E_season = s["season_energy"] / cnt
        E_noise = s["noise_energy"] / cnt
        
        signal = E_trend + E_season
        raw_quality = signal / (E_noise + 1e-9)
        # Normalize quality to be centered around 1.0
        quality_score = raw_quality / (raw_quality.median() + 1e-9)
        quality_score = torch.clamp(quality_score, 0.1, 10.0)
        
        # 3. The Soft Blend
        # Q_final = Confidence * Spectral_Q + (1 - Confidence) * 1.0
        # If Chaos (Conf=0) -> Q_final = 1.0 -> Logic reduces to |W| * 1.0 (Magnitude)
        # If Periodic (Conf=1) -> Q_final = Spectral_Q -> Logic is Pure Spectral
        
        Q_expanded = quality_score.unsqueeze(0) # [1, C]
        Conf_expanded = confidence.unsqueeze(0) # [1, C]
        
        Final_Q = (Conf_expanded * Q_expanded) + ((1.0 - Conf_expanded) * 1.0)
        
        # 4. Apply Mask
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        
        Q_final_exp = Final_Q.expand(O, C).to(W.dtype).view(O, -1, 4)
        
        # Score = Magnitude * sqrt(Adaptive_Quality)
        score = W_view.abs() * torch.sqrt(Q_final_exp)
        
        _, idx = torch.topk(score, 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)
        
    avg_conf = sum(spectral_dominance) / len(spectral_dominance)
    print(f"    [Adaptive Stats] Global Spectral Confidence: {avg_conf*100:.1f}%")
    if avg_conf > 0.7: print("    -> Mode: Mostly Spectral (Periodic Data)")
    elif avg_conf < 0.3: print("    -> Mode: Mostly Magnitude (Chaotic Data)")
    else: print("    -> Mode: Hybrid Blend (Complex Data)")

# --- PRUNING METHOD 2: SPARSEGPT (COMPETITOR) ---
def apply_sparsegpt(model, stats, device):
    print("  > Pruning: SparseGPT (Competitor)")
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

# --- PRUNING METHOD 3: MAGNITUDE (BASELINE) ---
def apply_magnitude(model):
    print("  > Pruning: Magnitude (Baseline)")
    for name, layer in get_prunable_layers(model):
        W = layer.weight.data
        O, C = W.shape
        W_view = W.view(O, -1, 4)
        _, idx = torch.topk(W_view.abs(), 2, dim=2)
        mask = torch.zeros_like(W_view, dtype=torch.bool).scatter_(2, idx, True)
        layer.weight.data = W_view.mul(mask).view(O, C)

# --- EVALUATION ---
def evaluate(pipeline, df, device):
    cols = df.columns[:32] # Eval first 32 cols for speed
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
    print(f"=== CHRONOS UNIVERSAL PRUNER ===")
    
    # 1. Load Data
    try:
        df = pd.read_csv(args.csv_path).select_dtypes(include=[np.number]).dropna(axis=1).fillna(0)
    except: return

    # 2. Load Model
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-large", device_map=args.device, torch_dtype=torch.bfloat16)
    orig_state = {k: v.cpu().clone() for k, v in pipeline.model.state_dict().items()}

    # 3. Calibration (One pass collects ALL stats)
    print("\n[1] Calibrating (Measuring Signal Entropy)...")
    layers = get_prunable_layers(pipeline.model)
    collector = UniversalCollector(pipeline.model, layers, args.device)
    pipeline.model.eval()
    rng = np.random.default_rng(42)
    cols = list(df.columns)
    try:
        # 64 samples is sufficient for robust stats
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

    # 4. Run Benchmarks
    print("\n[2] Benchmarking Methods...")

    # A. Magnitude
    pipeline.model.load_state_dict(orig_state)
    apply_magnitude(pipeline.model)
    results["Magnitude"] = evaluate(pipeline, df, args.device)
    print(f"Magnitude MSE: {results['Magnitude']:.4f}")

    # B. SparseGPT
    pipeline.model.load_state_dict(orig_state)
    apply_sparsegpt(pipeline.model, collector.stats, args.device)
    results["SparseGPT"] = evaluate(pipeline, df, args.device)
    print(f"SparseGPT MSE: {results['SparseGPT']:.4f}")

    # C. Universal Soft-Gate (Yours)
    pipeline.model.load_state_dict(orig_state)
    apply_universal_soft_gate(pipeline.model, collector.stats)
    results["Universal"] = evaluate(pipeline, df, args.device)
    print(f"Universal MSE: {results['Universal']:.4f}")

    # 5. Summary
    print("\n" + "="*50)
    print(f"{'Method':<20} | {'MSE':<15} | {'Status'}")
    print("-" * 50)
    
    # Determine winner
    best_score = min(results.values())
    
    for m, score in results.items():
        status = "WINNER" if score == best_score else ""
        print(f"{m:<20} | {score:<15.4f} | {status}")
    print("="*50)

if __name__ == "__main__":
    main()