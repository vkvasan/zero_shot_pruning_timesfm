#!/usr/bin/env python3
"""
snr_2of4_error_weighted_fair_v3.py

Fair + Transparent "Error-Weighted" calibration for strict 2:4 SNR pruning.

Key fixes vs the original quick script:
- Prints the *actual* split sizes and how many test windows are used (supports --test_windows -1 == all).
- Defines a calibration compute budget K = max_calls_per_layer * calib_batch (per layer),
  and makes window selection explicit via --calib_select {first,random,topk}.
- Avoids the batch-mismatch crash by allowing a separate --calib_batch (keep this small, e.g., 4).

Notes on fairness:
- This is *train-label-assisted calibration* when error_power>0 or calib_select=topk because it uses Y_calib
  to compute per-window error weights. Test labels are never used.

NEW in this version:
- --cuda_device (default=3): attempts to place TimesFM torch module on that GPU
- --prune_mode {snr,fused}: "fused" uses hidden-state spectral+moment stats to blend SNR/WANDA/MAG per layer
- --layer_report: writes per-layer hidden-state spectral stats and fused gate weights to CSV
"""

import argparse
import re
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def softmax3(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def gate_for_layer(st: "GramStat", season_strength: float, trend_strength: float, eps=1e-8, temp=1.0, floor=0.05):
    # kurtosis
    v = (st.m2 / max(st.n, 1)) + eps
    k_excess = (st.m4 / max(st.n, 1)) / (v * v) - 3.0

    # roughness
    r = (st.dx2 / max(st.ndx, 1)) / v if st.ndx > 0 else 0.0

    relu_k = max(0.0, k_excess)

    logit_snr   = +1.2 * season_strength + 0.6 * trend_strength - 0.8 * relu_k - 0.4 * r
    logit_wanda = +1.0 * relu_k + 0.3 * r + 0.3 * season_strength
    logit_mag   = 0.0

    w = softmax3([logit_snr / temp, logit_wanda / temp, logit_mag / temp])
    w = np.maximum(w, floor)
    w = w / w.sum()
    return float(w[0]), float(w[1]), float(w[2]), float(k_excess), float(r)


def forecast_timesfm_point(tfm_model, X: np.ndarray, horizon: int) -> np.ndarray:
    # TimesFM API expects list of arrays
    inputs = [X[i].astype(np.float32) for i in range(X.shape[0])]
    point_forecast, _quant = tfm_model.forecast(horizon=horizon, inputs=inputs)
    return np.asarray(point_forecast, dtype=np.float32)


def timed_forecast(tfm_model, X: np.ndarray, horizon: int, batch: int):
    preds, times = [], []
    n = X.shape[0]
    # warmup
    for i in range(min(n, 2 * batch)):
        _ = forecast_timesfm_point(tfm_model, X[i:i+1], horizon=horizon)
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
        raise ValueError("No windows produced (check train_end/context/horizon/stride).")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def mse_mae(pred: np.ndarray, tgt: np.ndarray):
    d = pred - tgt
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))


# -------------------------
# Hidden-state spectral summary (from hook-collected FFT bands)
# -------------------------
def spectral_strengths_from_stat(st: "GramStat", eps: float = 1e-12):
    """
    Returns scalar summaries for the activation sequence entering a Linear:
      trend_frac, season_frac, noise_frac
      trend_strength, season_strength (log(signal/noise)-style)
      sfm_mean
    """
    if st.trend_energy is None or st.season_energy is None or st.noise_energy is None or st.count <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")
    cnt = float(max(st.count, 1))
    Et = float((st.trend_energy / cnt).sum().item())
    Es = float((st.season_energy / cnt).sum().item())
    En = float((st.noise_energy / cnt).sum().item())
    tot = Et + Es + En + eps
    trend_frac = Et / tot
    season_frac = Es / tot
    noise_frac = En / tot
    trend_strength = float(np.log((Et + eps) / (En + eps)))
    season_strength = float(np.log((Es + eps) / (En + eps)))
    if st.sfm_sum is not None:
        sfm_mean = float((st.sfm_sum / cnt).mean().item())
    else:
        sfm_mean = float("nan")
    return trend_frac, season_frac, noise_frac, trend_strength, season_strength, sfm_mean


# -------------------------
# Targets
# -------------------------
def select_linears(torch_mod: nn.Module,
                   include_quantile_head: bool,
                   include_regex: str,
                   exclude_regex: str):
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    out = []
    for name, m in torch_mod.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        nl = name.lower()
        if (not include_quantile_head) and ("output_projection_quantiles" in nl):
            continue
        if inc and not inc.match(name):
            continue
        if exc and exc.match(name):
            continue
        out.append((name, m))
    return out


# -------------------------
# Gram Collection (grouped 4)
# -------------------------
@dataclass
class GramStat:
    Gsig: torch.Tensor
    Csig: float
    Gnoi: Optional[torch.Tensor]
    Cnoi: float

    # NEW (for WANDA + gating)
    Gact: torch.Tensor          # unweighted gram [G,4,4]
    Cact: float
    m2: float                   # sum(x^2)
    m4: float                   # sum(x^4)
    n: int                      # count of samples used for moments
    dx2: float                  # sum((x_t-x_{t-1})^2) for temporal roughness
    ndx: int

    # Spectral Stats (Universal)
    sfm_sum: Optional[torch.Tensor] = None
    trend_energy: Optional[torch.Tensor] = None
    season_energy: Optional[torch.Tensor] = None
    noise_energy: Optional[torch.Tensor] = None
    count: int = 0


@torch.no_grad()
def prune_linear_fused_2of4(layer, Gs, Gn, Gact, gate, eps=1e-8):
    # gate = (alpha_snr, beta_wanda, gamma_mag)
    a, b, c = gate

    W = layer.weight.data
    Cin = int(W.shape[1])
    G = Cin // 4
    Cg = G * 4
    if Cg == 0:
        return

    ds = torch.diagonal(Gs, dim1=-2, dim2=-1).float()
    dn = torch.ones_like(ds) if Gn is None else torch.diagonal(Gn, dim1=-2, dim2=-1).float()
    da = torch.diagonal(Gact, dim1=-2, dim2=-1).float()

    ds = ds.clamp_min(0.0) + eps
    dn = dn.clamp_min(0.0) + eps
    da = da.clamp_min(0.0) + eps

    ds = ds.to(W.device, dtype=torch.float32)
    dn = dn.to(W.device, dtype=torch.float32)
    da = da.to(W.device, dtype=torch.float32)

    scale_snr   = torch.sqrt(ds / dn)      # [G,4]
    scale_wanda = torch.sqrt(da)           # [G,4]

    Wf = W[:, :Cg].float().view(W.shape[0], G, 4)

    s_mag   = Wf.abs()
    s_wanda = Wf.abs() * scale_wanda.unsqueeze(0)
    s_snr   = Wf.abs() * scale_snr.unsqueeze(0)

    # Optional normalization so one expert doesn't dominate by scale alone
    def norm4(s):
        denom = (s.mean(dim=2, keepdim=True) + 1e-8)
        return s / denom

    s_mag, s_wanda, s_snr = norm4(s_mag), norm4(s_wanda), norm4(s_snr)
    scores = a * s_snr + b * s_wanda + c * s_mag

    top2 = torch.topk(scores, k=2, dim=2, largest=True).indices
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask.scatter_(2, top2, True)

    Wf = Wf * mask.to(Wf.dtype)
    W[:, :Cg] = Wf.reshape(W.shape[0], Cg).to(W.dtype)
    if Cg < Cin:
        W[:, Cg:] = 0


@torch.no_grad()
def compute_errors_and_weights(
    tfm_model,
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    horizon: int,
    calib_batch: int,
    error_power: float,
):
    """
    Returns:
      errors:    [N] MSE per window
      err_ratio: [N] errors / mean(errors)
      weights:   [N] normalized weights = (err_ratio)^error_power
    """
    preds = []
    for i in range(0, len(X_pool), calib_batch):
        preds.append(forecast_timesfm_point(tfm_model, X_pool[i:i+calib_batch], horizon))
    preds = np.concatenate(preds, axis=0)  # [N, H]
    diff = preds - Y_pool
    errors = np.mean(diff**2, axis=1).astype(np.float32)
    mean_err = float(np.mean(errors))
    err_ratio = (errors / (mean_err + 1e-6)).astype(np.float32)
    weights = (err_ratio) ** float(error_power)
    weights = weights.astype(np.float32)
    return errors, err_ratio, weights


@torch.no_grad()
def collect_group_grams_signal_noise(
    tfm_model,
    targets: List[Tuple[str, nn.Linear]],
    X_sel: np.ndarray,
    w_sig_sel: np.ndarray,            # [K]
    w_noi_sel: Optional[np.ndarray],  # [K] or None
    horizon: int,
    calib_batch: int,
    sample_rows_per_call: int,
    max_calls_per_layer: int,
):
    """
    Collect per-layer grouped (4) covariance grams.

    For each target Linear layer we estimate:
      Gsig = sum_n w_sig[n] * x_n x_n^T
      Gnoi = sum_n w_noi[n] * x_n x_n^T    (optional)

    Notes:
    - All weights are derived from TRAIN calibration windows only.
    - Uses forward pre-hooks on nn.Linear modules.
    - Keep calib_batch small enough that TimesFM doesn't internally split it.
    """
    stats: Dict[str, GramStat] = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []

    global current_batch_wsig, current_batch_wnoi
    current_batch_wsig = None
    current_batch_wnoi = None
    warned_split = False

    def make_hook(layer_name: str):
        def pre_hook(_mod, inputs):
            nonlocal warned_split
            if calls[layer_name] >= max_calls_per_layer:
                return

            (x,) = inputs
            B = x.shape[0]

            if current_batch_wsig is None:
                raise RuntimeError("Internal error: current_batch_wsig not set.")

            ws_batch = current_batch_wsig.to(x.device)
            wn_batch = None if current_batch_wnoi is None else current_batch_wnoi.to(x.device)

            # If TimesFM internally micro-batches, B may differ
            if ws_batch.numel() != B:
                if not warned_split:
                    print(
                        f"[warn] calib batch mismatch inside model: weights={ws_batch.numel()} "
                        f"but hook sees B={B}. TimesFM likely micro-batched. "
                        f"Set --calib_batch smaller (e.g., 4 or 1)."
                    )
                    warned_split = True

                if ws_batch.numel() >= B:
                    ws_batch = ws_batch[:B]
                    if wn_batch is not None:
                        if wn_batch.numel() >= B:
                            wn_batch = wn_batch[:B]
                        else:
                            wn_batch = torch.cat(
                                [wn_batch, wn_batch.new_full((B - wn_batch.numel(),), float(wn_batch[-1].item()))], dim=0
                            )
                else:
                    pad = ws_batch.new_full((B - ws_batch.numel(),), float(ws_batch[-1].item()))
                    ws_batch = torch.cat([ws_batch, pad], dim=0)
                    if wn_batch is not None:
                        padn = wn_batch.new_full((B - wn_batch.numel(),), float(wn_batch[-1].item()))
                        wn_batch = torch.cat([wn_batch, padn], dim=0)

            # x shape: [B, T, C] or [B, C]
            if x.dim() == 3:
                T = x.shape[1]
                xf = x.reshape(-1, x.shape[-1])  # [B*T, C]
                ws_exp = ws_batch.unsqueeze(1).expand(B, T).reshape(-1)
                wn_exp = None if wn_batch is None else wn_batch.unsqueeze(1).expand(B, T).reshape(-1)
            else:
                xf = x
                ws_exp = ws_batch
                wn_exp = wn_batch

            G = xf.shape[-1] // 4
            Cg = G * 4
            if Cg == 0:
                return

            # Moments & roughness on full data (sliced to Cg)
            vals = xf[:, :Cg]
            m2 = float((vals * vals).sum().item())
            m4 = float(((vals * vals) ** 2).sum().item())
            n_m = int(vals.numel())

            dx2 = 0.0
            ndx = 0
            if x.dim() == 3 and x.shape[1] > 1:
                x3 = x[:, :, :Cg].float()
                dx = x3[:, 1:, :] - x3[:, :-1, :]
                flat = dx.reshape(-1)
                if flat.numel() > sample_rows_per_call * 4:
                    idx = torch.randint(0, flat.numel(), (sample_rows_per_call * 4,), device=flat.device)
                    flat = flat.index_select(0, idx)
                dx2 = float((flat * flat).sum().item())
                ndx = int(flat.numel())

            # Spectral Analysis (Universal)
            sfm_batch = None
            trend_batch = None
            season_batch = None
            noise_batch = None

            if x.dim() == 3:
                x_float = x.float()
                x_fft = torch.fft.rfft(x_float, dim=1)
                energy = x_fft.abs().pow(2)

                # Spectral flatness measure
                psd = energy + 1e-12
                geo_mean = torch.exp(torch.mean(torch.log(psd), dim=1))
                ari_mean = torch.mean(psd, dim=1)
                sfm = geo_mean / (ari_mean + 1e-12)
                sfm_batch = sfm.sum(dim=0).detach().cpu()  # [C]

                # Energy bands
                freq_len = energy.shape[1]
                idx_trend = max(1, int(freq_len * 0.05))
                idx_noise = int(freq_len * 0.70)

                trend = energy[:, :idx_trend, :].sum(dim=(0, 1)).detach().cpu()
                noise = energy[:, idx_noise:, :].sum(dim=(0, 1)).detach().cpu()

                mid_band = energy[:, idx_trend:idx_noise, :]
                if mid_band.shape[1] > 0:
                    top_k_vals, _ = torch.topk(mid_band, k=min(3, mid_band.shape[1]), dim=1)
                    season = top_k_vals.sum(dim=(0, 1)).detach().cpu()
                else:
                    season = torch.zeros_like(trend)

                trend_batch = trend
                season_batch = season
                noise_batch = noise

            # Subsampling for grams
            xf = vals
            if xf.shape[0] > sample_rows_per_call:
                idx = torch.randint(0, xf.shape[0], (sample_rows_per_call,), device=xf.device)
                xf = xf.index_select(0, idx)
                ws_exp = ws_exp.index_select(0, idx)
                if wn_exp is not None:
                    wn_exp = wn_exp.index_select(0, idx)

            xg = xf.reshape(xf.shape[0], G, 4)  # [N,G,4]

            # Grams
            wu = torch.ones_like(ws_exp)
            Gact_batch = torch.einsum("n,ngc,ngd->gcd", wu, xg, xg).detach().cpu()
            Cact = float(wu.sum().item())

            Gs_batch = torch.einsum("n,ngc,ngd->gcd", ws_exp, xg, xg).detach().cpu()
            Cs = float(ws_exp.sum().item())

            Gn_batch = None
            Cn = 0.0
            if wn_exp is not None:
                Gn_batch = torch.einsum("n,ngc,ngd->gcd", wn_exp, xg, xg).detach().cpu()
                Cn = float(wn_exp.sum().item())

            if layer_name not in stats:
                stats[layer_name] = GramStat(
                    Gsig=Gs_batch, Csig=Cs,
                    Gnoi=Gn_batch, Cnoi=Cn,
                    Gact=Gact_batch, Cact=Cact,
                    m2=m2, m4=m4, n=n_m,
                    dx2=dx2, ndx=ndx,
                    sfm_sum=sfm_batch,
                    trend_energy=trend_batch,
                    season_energy=season_batch,
                    noise_energy=noise_batch,
                    count=x.shape[0] if x.dim() == 3 else 0,
                )
            else:
                st = stats[layer_name]
                st.Gsig += Gs_batch
                st.Csig += Cs

                if wn_exp is not None:
                    if st.Gnoi is None:
                        st.Gnoi = Gn_batch
                        st.Cnoi = Cn
                    else:
                        st.Gnoi += Gn_batch
                        st.Cnoi += Cn

                st.Gact += Gact_batch
                st.Cact += Cact
                st.m2 += m2
                st.m4 += m4
                st.n += n_m
                st.dx2 += dx2
                st.ndx += ndx

                if sfm_batch is not None:
                    if st.sfm_sum is None:
                        st.sfm_sum = sfm_batch
                        st.trend_energy = trend_batch
                        st.season_energy = season_batch
                        st.noise_energy = noise_batch
                        st.count = x.shape[0]
                    else:
                        st.sfm_sum += sfm_batch
                        st.trend_energy += trend_batch
                        st.season_energy += season_batch
                        st.noise_energy += noise_batch
                        st.count += x.shape[0]

            calls[layer_name] += 1

        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    # Drive forward passes (for hooks)
    n = X_sel.shape[0]
    for i in range(0, n, calib_batch):
        end = min(i + calib_batch, n)
        current_batch_wsig = torch.from_numpy(w_sig_sel[i:end]).float()
        current_batch_wnoi = None if w_noi_sel is None else torch.from_numpy(w_noi_sel[i:end]).float()
        xb = X_sel[i:end]
        _ = forecast_timesfm_point(tfm_model, xb, horizon=horizon)

    for h in hooks:
        h.remove()

    return stats


# -------------------------
# SNR Pruning (2:4)
# -------------------------
PAIRS = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.long)
PAIR_MASKS = torch.zeros((6,4), dtype=torch.float32)
for k in range(6):
    i, j = PAIRS[k].tolist()
    PAIR_MASKS[k, i] = 1.0
    PAIR_MASKS[k, j] = 1.0


@torch.no_grad()
def prune_linear_snr_2of4(layer, st: GramStat, score_mode: str, eps: float, refit: bool, ridge: float):
    """Strict 2:4 pruning for a Linear layer.

    score_mode:
      - "keep": keep-energy only
      - "ratio": keep/drop ratio on a single gram
      - "sn_ratio2": ratio-of-ratios using signal/noise grams (forward-only)
      - "spectral": spectral-only scoring
      - "unified": noise-aware blend of ratio + spectral
    """
    W = layer.weight.data
    O, C = W.shape
    Ggroups = C // 4
    Cg = Ggroups * 4
    if Cg == 0:
        return

    device = W.device
    dtype = W.dtype

    Wg = W[:, :Cg].view(O, Ggroups, 4)
    Gs = (st.Gsig / max(st.Csig, 1e-6)).to(device=device, dtype=dtype)

    use_noise = (score_mode == "sn_ratio2") and (st.Gnoi is not None) and (st.Cnoi > 0.0)
    if use_noise:
        Gn = (st.Gnoi / max(st.Cnoi, 1e-6)).to(device=device, dtype=dtype)
    else:
        Gn = None
        if score_mode == "sn_ratio2":
            score_mode = "ratio"

    masks = PAIR_MASKS.to(device=device, dtype=dtype)
    scores = torch.empty((O, Ggroups, 6), device=device, dtype=dtype)

    for k in range(6):
        mk = masks[k].view(1, 1, 4)
        md = 1.0 - mk
        Wk = Wg * mk
        Wd = Wg * md

        # signal energies
        Tk_s = torch.einsum("ogc,gcd->ogd", Wk, Gs)
        Ek_s = (Tk_s * Wk).sum(dim=2)
        Td_s = torch.einsum("ogc,gcd->ogd", Wd, Gs)
        Ed_s = (Td_s * Wd).sum(dim=2)

        if score_mode == "keep":
            scores[:, :, k] = Ek_s

        elif score_mode == "ratio":
            scores[:, :, k] = Ek_s / (Ed_s + eps)

        elif score_mode == "spectral":
            if st.sfm_sum is None or st.count == 0:
                scores[:, :, k] = Ek_s / (Ed_s + eps)
            else:
                cnt = float(max(st.count, 1))
                sfm_avg = (st.sfm_sum / cnt).to(device=device, dtype=dtype)
                E_trend = (st.trend_energy / cnt).to(device=device, dtype=dtype)
                E_season = (st.season_energy / cnt).to(device=device, dtype=dtype)
                E_noise = (st.noise_energy / cnt).to(device=device, dtype=dtype)

                confidence = 1.0 - torch.clamp((sfm_avg - 0.2) / 0.4, 0.0, 1.0)
                signal = E_trend + E_season
                raw_quality = signal / (E_noise + 1e-9)
                quality_score = raw_quality / (raw_quality.median() + 1e-9)
                quality_score = torch.clamp(quality_score, 0.1, 10.0)

                Q_final = (confidence * quality_score) + ((1.0 - confidence) * 1.0)
                Q_view = Q_final[:Cg].view(1, Ggroups, 4).expand(O, Ggroups, 4)
                sqrt_Q = torch.sqrt(Q_view)
                Score_k = (Wk.abs() * sqrt_Q).sum(dim=2)
                scores[:, :, k] = Score_k

        elif score_mode == "unified":
            S_ratio = Ek_s / (Ed_s + eps)

            if st.sfm_sum is None or st.count == 0:
                scores[:, :, k] = S_ratio
            else:
                cnt = float(max(st.count, 1))
                sfm_avg = (st.sfm_sum / cnt).to(device=device, dtype=dtype)
                E_trend = (st.trend_energy / cnt).to(device=device, dtype=dtype)
                E_season = (st.season_energy / cnt).to(device=device, dtype=dtype)
                E_noise = (st.noise_energy / cnt).to(device=device, dtype=dtype)

                total_E = E_trend + E_season + E_noise + 1e-12
                noise_frac = E_noise / total_E
                noise_frac_scalar = float(noise_frac.mean().item())

                # Alpha: noise_frac < 0.15 => 0 (pure ratio), >0.30 => 1 (pure spectral)
                alpha = max(0.0, min(1.0, (noise_frac_scalar - 0.15) / 0.15))

                if alpha < 0.01:
                    scores[:, :, k] = S_ratio
                else:
                    confidence = 1.0 - torch.clamp((sfm_avg - 0.2) / 0.4, 0.0, 1.0)
                    signal = E_trend + E_season
                    raw_quality = signal / (E_noise + 1e-9)
                    quality_score = raw_quality / (raw_quality.median() + 1e-9)
                    quality_score = torch.clamp(quality_score, 0.1, 10.0)

                    Q_final = (confidence * quality_score) + ((1.0 - confidence) * 1.0)
                    Q_view = Q_final[:Cg].view(1, Ggroups, 4).expand(O, Ggroups, 4)
                    sqrt_Q = torch.sqrt(Q_view)
                    S_spectral = (Wk.abs() * sqrt_Q).sum(dim=2)

                    def znorm(t):
                        mu = t.mean()
                        sd = t.std() + 1e-9
                        return (t - mu) / sd

                    S_ratio_n = znorm(S_ratio)
                    S_spectral_n = znorm(S_spectral)
                    scores[:, :, k] = alpha * S_spectral_n + (1.0 - alpha) * S_ratio_n

        else:
            # score_mode == "sn_ratio2"
            assert Gn is not None
            Tk_n = torch.einsum("ogc,gcd->ogd", Wk, Gn)
            Ek_n = (Tk_n * Wk).sum(dim=2)
            Td_n = torch.einsum("ogc,gcd->ogd", Wd, Gn)
            Ed_n = (Td_n * Wd).sum(dim=2)

            Rs = Ek_s / (Ed_s + eps)
            Rn = Ek_n / (Ed_n + eps)
            scores[:, :, k] = Rs / (Rn + eps)

    bestk = torch.argmax(scores, dim=2)
    pair_idx = PAIRS.to(device=device)[bestk]
    keep_mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool)
    keep_mask.scatter_(2, pair_idx, torch.ones_like(pair_idx, dtype=torch.bool))

    if not refit:
        Wnew = torch.where(keep_mask, Wg, torch.zeros_like(Wg))
    else:
        # Refit with signal gram
        B = torch.einsum("ogc,gcd->ogd", Wg, Gs)
        invs = []
        eye2 = torch.eye(2, device=device, dtype=dtype).view(1, 2, 2)
        for k in range(6):
            i, j = PAIRS[k].tolist()
            Gss = Gs[:, [i, j]][:, :, [i, j]] + ridge * eye2
            invs.append(torch.inverse(Gss))
        invs = torch.stack(invs, dim=0)  # [6,G,2,2]

        Wnew = torch.zeros_like(Wg)
        g_idx = torch.arange(Ggroups, device=device).view(1, Ggroups).expand(O, Ggroups)
        for k in range(6):
            sel = (bestk == k)
            if not torch.any(sel):
                continue
            i, j = PAIRS[k].tolist()
            b0 = B[:, :, i][sel]
            b1 = B[:, :, j][sel]
            bsel = torch.stack([b0, b1], dim=1)  # [N,2]
            gsel = g_idx[sel]
            invsel = invs[k, gsel, :, :]
            u = torch.bmm(invsel, bsel.unsqueeze(2)).squeeze(2)
            Wnew[:, :, i][sel] = u[:, 0]
            Wnew[:, :, j][sel] = u[:, 1]

    W[:, :Cg] = Wnew.view(O, Cg)
    if Cg < C:
        W[:, Cg:] = 0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--col", type=str, default="OT")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--train_end", type=int, default=49152)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_test", type=int, default=96)

    ap.add_argument("--calib_windows", type=int, default=1091, help="Calibration pool size (from start of train windows).")
    ap.add_argument("--calib_select", type=str, default="first", choices=["first", "random", "topk"],
                    help="Which K windows from the calib pool are used for gram collection (K is compute budget).")

    ap.add_argument("--test_windows", type=int, default=-1, help="-1 means use all available test windows.")
    ap.add_argument("--test_start_window", type=int, default=0)

    ap.add_argument("--batch", type=int, default=4, help="Evaluation (test) batch size.")
    ap.add_argument("--calib_batch", type=int, default=4, help="Calibration batch size for error + gram collection (keep small, e.g., 4).")

    ap.add_argument("--include_quantile_head", action="store_true")
    ap.add_argument("--include_regex", type=str, default=".*")
    ap.add_argument("--exclude_regex", type=str, default="")
    ap.add_argument("--sample_rows_per_call", type=int, default=2048)
    ap.add_argument("--max_calls_per_layer", type=int, default=32)

    ap.add_argument("--score_mode", type=str, default="sn_ratio2",
                    choices=["sn_ratio2", "ratio", "keep", "spectral", "unified"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-5)

    ap.add_argument("--model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    ap.add_argument("--cuda_device", type=int, default=3,
                    help="CUDA device index. Tip: use CUDA_VISIBLE_DEVICES=3 and then set --cuda_device 0.")

    ap.add_argument("--error_power", type=float, default=1.0,
                    help="Exponent for error weighting. Set 0 for uniform weights (label-free if calib_select != topk).")
    ap.add_argument("--sn_gamma", type=float, default=1.0,
                    help="For score_mode=sn_ratio2: build two grams using err_ratio^(-sn_gamma) as signal weights and "
                         "err_ratio^(+sn_gamma) as noise weights.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--measure_time", action="store_true")

    # New: hidden-state-informed fused pruning
    ap.add_argument("--prune_mode", type=str, default="snr", choices=["snr", "fused"],
                    help="snr = original prune_linear_snr_2of4; fused = hidden-state gated blend of SNR/WANDA/MAG.")
    ap.add_argument("--fusion_temp", type=float, default=1.0, help="Softmax temperature for fused gate weights.")
    ap.add_argument("--fusion_floor", type=float, default=0.05, help="Minimum floor per expert in fused gate.")
    ap.add_argument("--layer_report", type=str, default="",
                    help="Optional CSV path for per-layer hidden-state spectral/moment stats and gate weights.")

    args = ap.parse_args()

    # -------------------------
    # Device setup
    # -------------------------
    if torch.cuda.is_available():
        ndev = torch.cuda.device_count()
        if args.cuda_device < 0 or args.cuda_device >= ndev:
            raise RuntimeError(
                f"--cuda_device={args.cuda_device} but visible CUDA devices={ndev}. "
                f"Use nvidia-smi and/or CUDA_VISIBLE_DEVICES."
            )
        torch.cuda.set_device(args.cuda_device)
        device = torch.device(f"cuda:{args.cuda_device}")
        print(f"[device] using {device} | name={torch.cuda.get_device_name(args.cuda_device)}")
    else:
        device = torch.device("cpu")
        print("[device] CUDA not available; using CPU")

    # Data
    series = load_series(args.csv, args.col)
    n_total = len(series)

    X_train, Y_train = make_windows(series, 0, args.train_end, args.context, args.horizon, args.stride_train)
    X_pool = X_train[:args.calib_windows]
    Y_pool = Y_train[:args.calib_windows]

    X_test_all, Y_test_all = make_windows(series, args.train_end, n_total, args.context, args.horizon, args.stride_test)
    ts = args.test_start_window
    if args.test_windows < 0:
        te = X_test_all.shape[0]
    else:
        te = min(ts + args.test_windows, X_test_all.shape[0])
    X_test, Y_test = X_test_all[ts:te], Y_test_all[ts:te]

    # Logs
    print(f"[split] total_rows={n_total} train_end={args.train_end} context={args.context} horizon={args.horizon}")
    print(f"[train] windows={X_train.shape[0]} calib_pool={X_pool.shape[0]}")
    req_tw = args.test_windows
    print(f"[test]  available={X_test_all.shape[0]} requested={req_tw} start={ts} using={X_test.shape[0]} stride_test={args.stride_test}")

    gram_budget = args.max_calls_per_layer * args.calib_batch
    eff_K = min(X_pool.shape[0], gram_budget)
    print(
        f"[calib] eval_batch={args.batch} calib_batch={args.calib_batch} "
        f"gram_budget=max_calls_per_layer*calib_batch={args.max_calls_per_layer}*{args.calib_batch}={gram_budget} "
        f"=> effective_K={eff_K}"
    )

    # Model
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    torch_mod = find_torch_module(tfm)
    torch_mod.to(device)
    torch_mod.eval()

    targets = select_linears(torch_mod, args.include_quantile_head, args.include_regex, args.exclude_regex)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=max(args.horizon, 256)))

    # Baseline eval
    if args.measure_time:
        pred_base, tsec = timed_forecast(tfm, X_test, args.horizon, args.batch)
        mse_b, mae_b = mse_mae(pred_base, Y_test)
        print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f} | avg_batch_sec={tsec:.4f}")
    else:
        preds = []
        for i in range(0, len(X_test), args.batch):
            preds.append(forecast_timesfm_point(tfm, X_test[i:i+args.batch], args.horizon))
        pred_base = np.concatenate(preds, axis=0)
        mse_b, mae_b = mse_mae(pred_base, Y_test)
        print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f}")

    # Decide if we need error computation
    need_errors = (args.calib_select == "topk") or (args.error_power != 0.0) or (args.score_mode == "sn_ratio2")

    if need_errors:
        print(f"[calib] computing errors/weights on pool: power={args.error_power} (batched={args.calib_batch}) ...")
        errors, err_ratio, weights = compute_errors_and_weights(
            tfm_model=tfm,
            X_pool=X_pool,
            Y_pool=Y_pool,
            horizon=args.horizon,
            calib_batch=args.calib_batch,
            error_power=args.error_power,
        )
        print(
            f"[calib] weight_stats: min={weights.min():.4f} max={weights.max():.4f} "
            f"mean={weights.mean():.4f} (power={args.error_power})"
        )
    else:
        errors = None
        err_ratio = np.ones((X_pool.shape[0],), dtype=np.float32)
        weights = np.ones((X_pool.shape[0],), dtype=np.float32)
        print(f"[calib] uniform weights (no labels used): power=0, select={args.calib_select}")

    # Select K windows
    rng = np.random.default_rng(args.seed)
    K = eff_K

    if args.calib_select == "first":
        sel_idx = np.arange(K, dtype=np.int64)
    elif args.calib_select == "random":
        sel_idx = rng.choice(X_pool.shape[0], size=K, replace=False).astype(np.int64)
    elif args.calib_select == "topk":
        if errors is None:
            raise RuntimeError("topk selection requires errors, but errors were not computed.")
        sel_idx = np.argsort(errors)[-K:].astype(np.int64)
    else:
        raise ValueError(f"Unknown calib_select: {args.calib_select}")

    X_sel = X_pool[sel_idx]

    # Weights for gram collection
    w_sel = weights[sel_idx].astype(np.float32)
    er_sel = err_ratio[sel_idx].astype(np.float32)

    if args.score_mode == "sn_ratio2":
        ws = (er_sel ** (-float(args.sn_gamma))).astype(np.float32)
        wn = (er_sel ** ( float(args.sn_gamma))).astype(np.float32)
        ws = (ws / (ws.mean() + 1e-8)).astype(np.float32)
        wn = (wn / (wn.mean() + 1e-8)).astype(np.float32)
        w_sig_sel, w_noi_sel = ws, wn
        print(
            f"[calib] sn_ratio2 grams: sn_gamma={args.sn_gamma:g} | "
            f"sig_w(min/mean/max)=({ws.min():.3g}/{ws.mean():.3g}/{ws.max():.3g}) "
            f"noi_w(min/mean/max)=({wn.min():.3g}/{wn.mean():.3g}/{wn.max():.3g})"
        )
    else:
        w_sig_sel, w_noi_sel = w_sel, None

    print(
        f"[calib] collecting grams: select={args.calib_select} pool={X_pool.shape[0]} "
        f"K={K} (max_calls_per_layer={args.max_calls_per_layer})"
    )

    gram_stats = collect_group_grams_signal_noise(
        tfm_model=tfm,
        targets=targets,
        X_sel=X_sel,
        w_sig_sel=w_sig_sel,
        w_noi_sel=w_noi_sel,
        horizon=args.horizon,
        calib_batch=args.calib_batch,
        sample_rows_per_call=args.sample_rows_per_call,
        max_calls_per_layer=args.max_calls_per_layer,
    )
    print(f"[calib] collected grams for {len(gram_stats)}/{len(targets)} layers")

    # Optional layer report (hidden-state interpretability summary)
    if args.layer_report:
        rows = []
        for name, _layer in targets:
            st = gram_stats.get(name, None)
            if st is None or st.Csig <= 0.0:
                continue

            trend_frac, season_frac, noise_frac, trend_strength, season_strength, sfm_mean = spectral_strengths_from_stat(st)
            a_snr, b_wanda, c_mag, k_excess, rough = gate_for_layer(
                st, season_strength=season_strength, trend_strength=trend_strength,
                temp=args.fusion_temp, floor=args.fusion_floor
            )

            rows.append({
                "layer": name,
                "trend_frac": trend_frac,
                "season_frac": season_frac,
                "noise_frac": noise_frac,
                "trend_strength_log": trend_strength,
                "season_strength_log": season_strength,
                "sfm_mean": sfm_mean,
                "kurtosis_excess": k_excess,
                "roughness": rough,
                "gate_snr": a_snr,
                "gate_wanda": b_wanda,
                "gate_mag": c_mag,
                "Csig": float(st.Csig),
                "Cnoi": float(st.Cnoi),
                "Cact": float(st.Cact),
            })

        if rows:
            pd.DataFrame(rows).to_csv(args.layer_report, index=False)
            print(f"[report] wrote layer report: {args.layer_report} (rows={len(rows)})")
        else:
            print("[report] no rows to write (no gram stats collected?)")

    # Prune
    if args.prune_mode == "snr":
        print(f"[prune] SNR 2:4: score_mode={args.score_mode}, refit={bool(args.refit)} ridge={args.ridge:g}")
    else:
        print(
            f"[prune] FUSED 2:4 (hidden-state gated): temp={args.fusion_temp:g} "
            f"floor={args.fusion_floor:g} | (SNR/WANDA/MAG blend per layer)"
        )

    # Spectral diagnostics (for spectral/unified score_mode)
    if args.score_mode in ("spectral", "unified"):
        c_layers = []
        for name, _layer in targets:
            st = gram_stats.get(name, None)
            if st is None or st.Csig <= 0.0 or st.sfm_sum is None or st.count == 0:
                continue
            cnt = float(max(st.count, 1))
            sfm_avg = st.sfm_sum / cnt
            conf = 1.0 - torch.clamp((sfm_avg - 0.2) / 0.4, 0.0, 1.0)
            c_val = float(conf.mean().item())
            c_layers.append(c_val)

        if c_layers:
            import statistics
            if len(c_layers) > 1:
                print(
                    f"[diag] C_layer stats over {len(c_layers)} layers: "
                    f"min={min(c_layers):.4f} max={max(c_layers):.4f} "
                    f"mean={statistics.mean(c_layers):.4f} median={statistics.median(c_layers):.4f} "
                    f"stdev={statistics.stdev(c_layers):.4f}"
                )
            else:
                print(f"[diag] C_layer: {c_layers[0]:.4f}")

            st0 = gram_stats.get(targets[0][0], None) if targets else None
            if st0 is not None and st0.sfm_sum is not None:
                cnt = float(max(st0.count, 1))
                E_t = float((st0.trend_energy / cnt).sum().item())
                E_s = float((st0.season_energy / cnt).sum().item())
                E_n = float((st0.noise_energy / cnt).sum().item())
                total = E_t + E_s + E_n + 1e-12
                print(f"[diag] Energy bands (layer0): trend={E_t/total:.1%} season={E_s/total:.1%} noise={E_n/total:.1%}")

    # Apply pruning
    for name, layer in targets:
        st = gram_stats.get(name, None)
        if st is None or st.Csig <= 0.0:
            continue

        if args.prune_mode == "snr":
            prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), args.ridge)
        else:
            # Hidden-state-derived strengths (trend/season vs noise)
            _tf, _sf, _nf, trend_strength, season_strength, _sfm = spectral_strengths_from_stat(st)

            gate = gate_for_layer(
                st,
                season_strength=season_strength,
                trend_strength=trend_strength,
                temp=args.fusion_temp,
                floor=args.fusion_floor,
            )[:3]  # (alpha_snr, beta_wanda, gamma_mag)

            Gs = st.Gsig / max(st.Csig, 1e-6)
            Gn = None if (st.Gnoi is None or st.Cnoi <= 0.0) else (st.Gnoi / max(st.Cnoi, 1e-6))
            Ga = st.Gact / max(st.Cact, 1e-6)

            prune_linear_fused_2of4(layer, Gs, Gn, Ga, gate, eps=1e-8)

    # Eval pruned
    if args.measure_time:
        pred_p, tsec2 = timed_forecast(tfm, X_test, args.horizon, args.batch)
        mse_p, mae_p = mse_mae(pred_p, Y_test)
        print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f} | avg_batch_sec={tsec2:.4f}")
    else:
        preds = []
        for i in range(0, len(X_test), args.batch):
            preds.append(forecast_timesfm_point(tfm, X_test[i:i+args.batch], args.horizon))
        pred_p = np.concatenate(preds, axis=0)
        mse_p, mae_p = mse_mae(pred_p, Y_test)
        print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f}")

    print(f"[delta] ΔMSE={(mse_p - mse_b):+.6f}  ΔMAE={(mae_p - mae_b):+.6f}")


if __name__ == "__main__":
    main()
