"""
prune_unified.py

True Competitive MoE Implementation.
Evaluates MAG, Wanda, SNR, and OBS experts per layer against held-out validation data.
Includes Refit Penalization and SNR Bias for robust generalization.
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
# TimesFM helpers
# -------------------------
def find_torch_module(obj) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    for attr in ("model", "_model", "module", "_module", "torch_model", "_torch_model"):
        m = getattr(obj, attr, None)
        if isinstance(m, nn.Module):
            return m
    for v in getattr(obj, "__dict__", {}).values():
        if isinstance(v, nn.Module):
            return v
    raise RuntimeError("Could not locate underlying torch nn.Module.")

def forecast_timesfm_point(tfm_model, X: np.ndarray, horizon: int) -> np.ndarray:
    inputs = [X[i].astype(np.float32) for i in range(X.shape[0])]
    point_forecast, _quant = tfm_model.forecast(horizon=horizon, inputs=inputs)
    return np.asarray(point_forecast, dtype=np.float32)

def timed_forecast(tfm_model, X: np.ndarray, horizon: int, batch: int):
    preds, times = [], []
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = X[i:i+batch]
        t0 = time.perf_counter()
        yb = forecast_timesfm_point(tfm_model, xb, horizon=horizon)
        t1 = time.perf_counter()
        preds.append(yb)
        times.append(t1 - t0)
    return np.concatenate(preds, axis=0), float(np.mean(times)) if times else 0.0

# -------------------------
# Data
# -------------------------
def load_series(csv_path: str, col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in {csv_path}.")
    return df[col].to_numpy(dtype=np.float32)

def make_windows(series: np.ndarray, start: int, end: int, context: int, horizon: int, stride: int):
    xs, ys = [], []
    last = end - (context + horizon)
    for i in range(start, last + 1, stride):
        xs.append(series[i:i+context])
        ys.append(series[i+context:i+context+horizon])
    if not xs:
        raise ValueError("No windows produced.")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)

def mse_mae(pred: np.ndarray, tgt: np.ndarray):
    d = pred - tgt
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))

# -------------------------
# Targets
# -------------------------
def select_linears(torch_mod: nn.Module,
                   include_quantile_head: bool,
                   include_regex: str):
    inc = re.compile(include_regex) if include_regex else None
    out = []
    for name, m in torch_mod.named_modules():
        if isinstance(m, nn.Linear):
            nl = name.lower()
            if (not include_quantile_head) and ("output_projection_quantiles" in nl):
                continue
            if inc and not inc.match(name):
                continue
            out.append((name, m))
    return out

# -------------------------
# Gram Collection
# -------------------------
@dataclass
class GramStat:
    Gsig: torch.Tensor
    Csig: float
    Gnoi: Optional[torch.Tensor]
    Cnoi: float
    Gact: torch.Tensor          
    Cact: float
    m2: float                   
    m4: float                   
    n: int                      
    trend_energy: Optional[torch.Tensor] = None
    season_energy: Optional[torch.Tensor] = None
    noise_energy: Optional[torch.Tensor] = None
    count: int = 0
    X_val: Optional[torch.Tensor] = None        
    avg_nsr: float = 1.0 # noise-to-signal ratio




@torch.no_grad()
def collect_stats(tfm_model, targets, X_sel, w_sig_sel, w_noi_sel, horizon, calib_batch, max_calls_per_layer):
    stats = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []
    
    global current_w_sig, current_w_noi
    current_w_sig = None
    current_w_noi = None

    def make_hook(name):
        def pre_hook(_mod, inputs):
            if calls[name] >= max_calls_per_layer: return
            (x,) = inputs
            B = x.shape[0]
            ws = current_w_sig.to(x.device) if current_w_sig is not None else torch.ones(B, device=x.device)
            wn = current_w_noi.to(x.device) if current_w_noi is not None else None
            
            xf = x.reshape(-1, x.shape[-1])
            C = xf.shape[-1]
            G = C // 4
            Cg = G * 4
            if Cg == 0: return
            xg = xf[:, :Cg].reshape(-1, G, 4)

            # Pruning Phase: First 48 windows for Grams, next windows for Validation
            is_val_phase = (calls[name] >= 48)
            
            if not is_val_phase:
                # Pruning Phase: Update Grams
                Gact = torch.einsum("ngc,ngd->gcd", xg, xg).cpu()
                Gsig = torch.einsum("n,ngc,ngd->gcd", ws.repeat_interleave(xf.shape[0]//B), xg, xg).cpu()
                Gnoi = torch.einsum("n,ngc,ngd->gcd", wn.repeat_interleave(xf.shape[0]//B), xg, xg).cpu() if wn is not None else None
                
                # Moments
                m2 = float((xg**2).sum()); m4 = float((xg**4).sum())
                
                # Spectral
                x_f = x.float()
                xfft = torch.fft.rfft(x_f, dim=1)
                e = xfft.abs().pow(2)
                it, in_ = max(1, int(e.shape[1]*0.05)), int(e.shape[1]*0.7)
                t_e = e[:, :it, :].sum(dim=(0,1)).cpu()
                s_e = e[:, it:in_, :].sum(dim=(0,1)).cpu()
                n_e = e[:, in_:, :].sum(dim=(0,1)).cpu()

                if name not in stats:
                    stats[name] = GramStat(
                        Gsig=Gsig, Csig=float(ws.sum()), Gnoi=Gnoi, Cnoi=float(wn.sum()) if wn is not None else 0.0,
                        Gact=Gact, Cact=float(xg.shape[0]), m2=m2, m4=m4, n=int(xg.numel()),
                        trend_energy=t_e, season_energy=s_e, noise_energy=n_e, count=B
                    )
                else:
                    st = stats[name]
                    st.Gsig += Gsig; st.Csig += float(ws.sum())
                    st.Gact += Gact; st.Cact += float(xg.shape[0])
                    if Gnoi is not None: st.Gnoi += Gnoi; st.Cnoi += float(wn.sum())
                    st.m2 += m2; st.m4 += m4; st.n += int(xg.numel())
                    st.trend_energy += t_e; st.season_energy += s_e; st.noise_energy += n_e; st.count += B
            else:
                # Validation Phase: Capture held-out activations
                if stats[name].X_val is None or stats[name].X_val.shape[0] < (64 * 512):
                    new_x = x.reshape(-1, x.shape[-1]).cpu()
                    if stats[name].X_val is None: stats[name].X_val = new_x
                    else: stats[name].X_val = torch.cat([stats[name].X_val, new_x], dim=0)[:32768]

            calls[name] += 1
        return pre_hook


    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    for i in range(0, X_sel.shape[0], calib_batch):
        current_w_sig = torch.from_numpy(w_sig_sel[i:i+calib_batch])
        current_w_noi = torch.from_numpy(w_noi_sel[i:i+calib_batch]) if w_noi_sel is not None else None
        _ = forecast_timesfm_point(tfm_model, X_sel[i:i+calib_batch], horizon)

    for h in hooks: h.remove()
    return stats

# -------------------------
# Pruning
# -------------------------
PAIRS = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.long)
PAIR_MASKS = torch.zeros((6,4), dtype=torch.float32)
for k in range(6):
    i,j = PAIRS[k].tolist()
    PAIR_MASKS[k,i] = PAIR_MASKS[k,j] = 1.0

@torch.no_grad()
def prune_linear_snr_2of4(layer, st: GramStat, score_mode: str, eps: float, refit: bool, ridge: float, horizon: int = 96, nf_hi: float = 0.25):
    W = layer.weight.data
    O, C = W.shape
    Ggroups = C // 4
    Cg = Ggroups * 4
    if Cg == 0: return
    device = W.device
    dtype = W.dtype

    Wg = W[:, :Cg].view(O, Ggroups, 4)
    Gs = (st.Gsig / max(st.Csig, 1e-6)).to(device=device, dtype=dtype)
    Ga = (st.Gact / max(st.Cact, 1e-6)).to(device=device, dtype=dtype)
    Gn = (st.Gnoi / max(st.Cnoi, 1e-6)).to(device=device, dtype=dtype) if st.Gnoi is not None and st.Cnoi > 0 else None


    if score_mode == "unified":
        import math
        damp_s = 0.01 * torch.mean(torch.diagonal(Gs, dim1=1, dim2=2))
        Hinv_s = torch.inverse(Gs + damp_s * torch.eye(4, device=device, dtype=dtype).unsqueeze(0))
        diag_s = torch.diagonal(Hinv_s, dim1=1, dim2=2)

        damp_a = 0.01 * torch.mean(torch.diagonal(Ga, dim1=1, dim2=2))
        Hinv_a = torch.inverse(Ga + damp_a * torch.eye(4, device=device, dtype=dtype).unsqueeze(0))
        diag_a = torch.diagonal(Hinv_a, dim1=1, dim2=2)
        act_diag = torch.diagonal(Ga, dim1=1, dim2=2).clamp_min(1e-8)

        log_ridge = math.log10(max(ridge, 1e-10))
        t_ridge = max(0.0, min(1.0, (log_ridge - (-3.0)) / 1.0))
        w_obs, w_ratio, w_mag = 0.25+0.6*t_ridge, 0.5-0.4*t_ridge, 0.25-0.2*t_ridge

        def znorm(t): return (t - t.mean()) / (t.std() + 1e-9)
        masks_t = PAIR_MASKS.to(device=device, dtype=dtype)
        
        wanda_imp = (Wg ** 2) * act_diag.unsqueeze(0)
        obs_imp_a = (Wg ** 2) / (diag_a.unsqueeze(0) + 1e-10)
        
        scores_mag = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        scores_wanda = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        scores_obs = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        snr_ratio = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        snr_mag   = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        snr_obsig = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)

        for k in range(6):
            mk = masks_t[k].view(1, 1, 4)
            Wk, Wd = Wg * mk, Wg * (1.0 - mk)
            scores_mag[:,:,k] = Wk.abs().sum(dim=2)
            scores_wanda[:,:,k] = (wanda_imp * mk).sum(dim=2)
            scores_obs[:,:,k] = (obs_imp_a * mk).sum(dim=2)
            
            Tk_s, Td_s = torch.einsum("ogc,gcd->ogd", Wk, Gs), torch.einsum("ogc,gcd->ogd", Wd, Gs)
            s_ratio = (Tk_s * Wk).sum(dim=2) / ((Td_s * Wd).sum(dim=2) + eps)
            
            if Gn is not None:
                Tk_n, Td_n = torch.einsum("ogc,gcd->ogd", Wk, Gn), torch.einsum("ogc,gcd->ogd", Wd, Gn)
                n_ratio = (Tk_n * Wk).sum(dim=2) / ((Td_n * Wd).sum(dim=2) + eps)
                snr_ratio[:,:,k] = s_ratio / (n_ratio + eps)
            else:
                snr_ratio[:,:,k] = s_ratio
                
            snr_mag[:,:,k] = Wk.abs().sum(dim=2)
            snr_obsig[:,:,k] = (((Wk**2)/(diag_s.unsqueeze(0)+1e-10))*mk).sum(dim=2)

        scores_snr = (w_ratio*znorm(snr_ratio.reshape(-1,6)) + w_mag*znorm(snr_mag.reshape(-1,6)) + w_obs*znorm(snr_obsig.reshape(-1,6))).reshape(O, Ggroups, 6)
        expert_bestks = [torch.argmax(scores_mag,2), torch.argmax(scores_wanda,2), torch.argmax(scores_snr,2), torch.argmax(scores_obs,2)]

        expert_names = ["MAG", "Wanda", "SNR", "OBS"]

        X_val_d = st.X_val.to(device=device, dtype=dtype) if st.X_val is not None else torch.randn((1, Cg), device=device, dtype=dtype)
        Y_val_dense = (X_val_d @ W[:,:Cg].T).detach()

        H_reg = Gs + ridge * torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        B = torch.einsum("ogc,gcd->ogd", Wg, H_reg)
        invs = torch.stack([torch.inverse(Gs[:,PAIRS[k]][:,:,PAIRS[k]] + ridge*torch.eye(2, device=device)) for k in range(6)])
        g_idx = torch.arange(Ggroups, device=device).view(1, Ggroups).expand(O, Ggroups)

        # Baseline MSE
        mse_dense = torch.mean(Y_val_dense**2).item() + 1e-10

        # Expert Lockdown (NSR Guard):
        # On extremely clean, seasonal datasets, local reconstruction metrics can be biased.
        # We lock in the robust SNR expert to preserve periodic structure.
        # Threshold lowered to 0.05 to only catch 'purest' cases (v13 restoration).
        if st.avg_nsr < 0.05:
            winner_bestk = expert_bestks[2] # SNR
            Wnew = torch.where(torch.zeros((O,Ggroups,4), device=device, dtype=torch.bool).scatter_(2, PAIRS.to(device)[winner_bestk], True), Wg, torch.zeros_like(Wg))
            import sys
            sys.stderr.write(f"[moe] Winner LOCK: SNR Refit=False (NSR={st.avg_nsr:.4f})\n")
        else:
            # TRUE Competitive MoE for restoration baseline
            refit_p  = 1.0
            SNR_BIAS = 1.0
            
            mask_mses = []
            for idx_e, bbestk in enumerate(expert_bestks):
                mask = torch.zeros((O,Ggroups,4), device=device, dtype=torch.bool).scatter_(2, PAIRS.to(device)[bbestk], True)
                W_m = torch.where(mask, Wg, torch.zeros_like(Wg))
                mse_m = torch.mean(((X_val_d @ W_m.view(O, Cg).T) - Y_val_dense)**2).item()
                mask_mses.append(mse_m)

            best_mask_mse = min(mask_mses)
            best_mask_idx = mask_mses.index(best_mask_mse)
            drop_frac = best_mask_mse / mse_dense
            best_mse, final_expert, final_refit = best_mask_mse, expert_names[best_mask_idx], False

            if refit:
                for idx_e, bbestk in enumerate(expert_bestks):
                    W_r = torch.zeros_like(Wg)
                    for k in range(6):
                        sel = (bbestk == k)
                        if not torch.any(sel): continue
                        bsel = torch.stack([B[:,:,PAIRS[k,0]][sel], B[:,:,PAIRS[k,1]][sel]], 1)
                        u = torch.bmm(invs[k, g_idx[sel]], bsel.unsqueeze(2)).squeeze(2)
                        W_r[:,:,PAIRS[k,0]][sel], W_r[:,:,PAIRS[k,1]][sel] = u[:, 0], u[:, 1]
                    mse_r = torch.mean(((X_val_d @ W_r.view(O, Cg).T) - Y_val_dense)**2).item()
                    if (mse_r * refit_p) < best_mask_mse:
                        if mse_r < best_mse:
                            best_mse, final_expert, final_refit = mse_r, expert_names[idx_e], True

            # SNR Robustness Bias
            snr_mse = mask_mses[2]
            if best_mse > (snr_mse * SNR_BIAS):
                best_mse, final_expert, final_refit = snr_mse, "SNR", False

            # Apply winner
            winner_bestk = expert_bestks[expert_names.index(final_expert)]
            if not final_refit:
                Wnew = torch.where(torch.zeros((O,Ggroups,4), device=device, dtype=torch.bool).scatter_(2, PAIRS.to(device)[winner_bestk], True), Wg, torch.zeros_like(Wg))
            else:
                Wnew = torch.zeros_like(Wg)
                for k in range(6):
                    sel = (winner_bestk == k)
                    if not torch.any(sel): continue
                    bsel = torch.stack([B[:,:,PAIRS[k,0]][sel], B[:,:,PAIRS[k,1]][sel]], 1)
                    u = torch.bmm(invs[k, g_idx[sel]], bsel.unsqueeze(2)).squeeze(2)
                    Wnew[:,:,PAIRS[k,0]][sel], Wnew[:,:,PAIRS[k,1]][sel] = u[:, 0], u[:, 1]
            import sys
            sys.stderr.write(f"[moe] Winner: {final_expert} Refit={final_refit} MSE_val={best_mse:.6f} drop={drop_frac:.4f}\n")


    else:
        top2 = torch.topk(Wg.abs(), 2, dim=2).indices
        Wnew = torch.where(torch.zeros_like(Wg, dtype=torch.bool).scatter_(2, top2, True), Wg, torch.zeros_like(Wg))

    W[:, :Cg] = Wnew.view(O, Cg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", default="OT")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--train_end", type=int, default=49152)
    ap.add_argument("--stride_test", type=int, default=96)
    ap.add_argument("--score_mode", default="unified")
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--max_calls_per_layer", type=int, default=64)
    ap.add_argument("--calib_batch", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--calib_select", default="last")
    ap.add_argument("--error_power", type=float, default=0.0)
    ap.add_argument("--nf_hi", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    torch_mod = find_torch_module(tfm)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=args.horizon))

    series = load_series(args.csv, args.col)
    X_train, Y_train = make_windows(series, 0, args.train_end, args.context, args.horizon, 1)
    X_test, Y_test = make_windows(series, args.train_end, len(series), args.context, args.horizon, args.stride_test)
    
    # Baseline
    preds_b = []
    for i in range(0, len(X_test), args.batch):
        preds_b.append(forecast_timesfm_point(tfm, X_test[i:i+args.batch], args.horizon))
    mse_b, mae_b = mse_mae(np.concatenate(preds_b, 0), Y_test)
    print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f}")

    # Stats
    X_pool = X_train[-1024:] if args.calib_select == "last" else X_train[:1024]
    Y_pool = Y_train[-1024:] if args.calib_select == "last" else Y_train[:1024]
    
    preds_pool = []
    for i in range(0, len(X_pool), args.calib_batch):
        preds_pool.append(forecast_timesfm_point(tfm, X_pool[i:i+args.calib_batch], args.horizon))
    errs = np.mean((np.concatenate(preds_pool, 0) - Y_pool)**2, axis=1)
    weights_sig = (errs / (errs.mean() + 1e-7))**args.error_power
    weights_noi = (1.0 / (errs + 1e-7)) / ((1.0 / (errs + 1e-7)).mean() + 1e-7)
    weights_noi = weights_noi**args.error_power # or some other scaling
    
    targets = select_linears(torch_mod, False, ".*")
    stats = collect_stats(tfm, targets, X_pool, weights_sig, weights_noi, args.horizon, args.calib_batch, args.max_calls_per_layer)

    # Spectral Diag
    names = list(stats.keys())
    if names:
        st0 = stats[names[0]]
        t0, s0, n0 = float(st0.trend_energy.sum()), float(st0.season_energy.sum()), float(st0.noise_energy.sum())
        total = t0 + s0 + n0 + 1e-9
        print(f"[diag] Energy bands ({names[0]}): trend={100*t0/total:.1f}% season={100*s0/total:.1f}% noise={100*n0/total:.1f}%")
        
        # Calculate avg fractions across all layers
        nsrs = []
        for name, st in stats.items():
            te, se, ne = float(st.trend_energy.sum()), float(st.season_energy.sum()), float(st.noise_energy.sum())
            # NSR = Noise / (Trend + Season)
            nsrs.append(ne / (te + se + 1e-9))
        avg_nsr = float(np.mean(nsrs))
        print(f"[diag] Average dataset noise-to-signal ratio (NSR): {avg_nsr:.4f}")
        # Add to GramStat for pruning function
        for st in stats.values():
            st.avg_nsr = avg_nsr



    # Prune

    for name, layer in targets:
        st = stats.get(name)
        if st: prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), args.ridge, horizon=args.horizon, nf_hi=args.nf_hi)

    # Eval
    preds_p = []
    for i in range(0, len(X_test), args.batch):
        preds_p.append(forecast_timesfm_point(tfm, X_test[i:i+args.batch], args.horizon))
    mse_p, mae_p = mse_mae(np.concatenate(preds_p, 0), Y_test)
    print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f}")
    print(f"[delta] ΔMSE={mse_p - mse_b:+.6f}")

if __name__ == "__main__":
    main()
