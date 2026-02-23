#!/usr/bin/env python3
"""
chronos2_snr_2of4_error_weighted.py

Chronos-2 strict 2:4 pruning using signal/noise grams (error-weighted).

Fixes vs your current broken version:
- Chronos-2 predict() returns QUANTILES (often 21). We convert to point forecast by taking median quantile.
- Handles fp16 safely: cast activations + weights to fp32 before einsum.
- Handles Chronos-2 internal token/patch batching: repeats per-window weights to match hook's B.

Prints BOTH normalized and raw metrics + raw deltas.

Example (ETTh2):
python chronos2_snr_2of4_error_weighted.py \
  --csv ETDataset/ETT-small/ETTh2.csv --cols_regex '^OT$' \
  --context 1024 --horizon 96 --train_end 8640 \
  --stride_train 1 --stride_test 96 \
  --calib_windows 1091 --calib_select first \
  --calib_batch 1 --max_calls_per_layer 256 --sample_rows_per_call 2048 \
  --model_id amazon/chronos-2 --device cuda --torch_dtype float16 \
  --num_samples 128 --batch 4 --measure_time \
  --score_mode ratio --sn_gamma 1.0 --error_power 1.0 \
  --include_regex '.*' \
  --exclude_regex '.*(embed|embedding|patch|token|input|pos|lm_head|head|output).*'
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


# -------------------------
# Metrics / utils
# -------------------------

def mse_mae(pred: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = pred - y
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Chronos loading
# -------------------------

def load_chronos(model_id: str, device: str, torch_dtype: str):
    dtype = None
    if torch_dtype and torch_dtype.lower() != "none":
        mp = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = mp[torch_dtype.lower()]

    # Prefer Chronos2Pipeline if present
    try:
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        return pipe, "chronos2"
    except Exception:
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        return pipe, "chronos"


# -------------------------
# Forecast conversion (IMPORTANT)
# -------------------------

def _to_point_forecast(t: torch.Tensor, horizon: int) -> np.ndarray:
    """
    Chronos-2 commonly returns quantiles: shape like [1, 21, H] or [21, H].
    We convert to point forecast by taking the MEDIAN quantile (middle index).
    """
    t = t.detach().cpu()

    # strip leading singleton dims
    while t.dim() > 1 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.dim() == 1:
        tt = t.flatten()
        if tt.numel() < horizon:
            pad = tt[-1].repeat(horizon - tt.numel())
            tt = torch.cat([tt, pad], dim=0)
        return tt[:horizon].numpy()

    if t.dim() == 2:
        # assume [Q, H] (quantiles) OR [S, H] (samples)
        if t.shape[1] != horizon and t.shape[0] == horizon:
            t = t.t()
        mid = t.shape[0] // 2
        return t[mid, :horizon].numpy()

    # if there are extra dims, peel until 2D-ish
    while t.dim() > 2:
        t = t[0]
    return _to_point_forecast(t, horizon)


@torch.no_grad()
def forecast_batch(pipe, pipe_kind: str, X: np.ndarray, horizon: int, batch_size: int, num_samples: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    inputs = [X[i].astype(np.float32, copy=False) for i in range(X.shape[0])]

    if pipe_kind == "chronos2":
        outs = pipe.predict(inputs, prediction_length=horizon, batch_size=batch_size)
        preds = [_to_point_forecast(t, horizon) for t in outs]
        return np.stack(preds, axis=0)

    # Chronos (v1) sometimes supports num_samples
    try:
        outs = pipe.predict(inputs, prediction_length=horizon, num_samples=num_samples, batch_size=batch_size)
    except TypeError:
        outs = pipe.predict(inputs, prediction_length=horizon, batch_size=batch_size)

    if isinstance(outs, list):
        preds = [_to_point_forecast(t, horizon) for t in outs]
        return np.stack(preds, axis=0)
    return _to_point_forecast(outs, horizon)[None, :]


@torch.no_grad()
def eval_model(pipe, pipe_kind: str, X: np.ndarray, Y: np.ndarray, horizon: int, batch: int, num_samples: int) -> Dict[str, float]:
    n = X.shape[0]
    preds = []
    for i in range(0, n, batch):
        xb = X[i:i+batch]
        pb = forecast_batch(pipe, pipe_kind, xb, horizon, batch_size=batch, num_samples=num_samples)
        preds.append(pb)
    P = np.concatenate(preds, axis=0)
    mse, mae = mse_mae(P, Y)
    return {"mse": mse, "mae": mae}


def timed_eval(pipe, pipe_kind: str, X: np.ndarray, Y: np.ndarray, horizon: int, batch: int, num_samples: int) -> Tuple[Dict[str, float], float]:
    _ = forecast_batch(pipe, pipe_kind, X[:1], horizon, batch_size=1, num_samples=num_samples)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = eval_model(pipe, pipe_kind, X, Y, horizon, batch, num_samples)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    nb = int(np.ceil(X.shape[0] / batch))
    return out, (t1 - t0) / max(1, nb)


# -------------------------
# Windowing + selection
# -------------------------

def build_windows(series: np.ndarray, context: int, horizon: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    T = series.shape[0]
    last_start = T - (context + horizon)
    if last_start < 0:
        return np.zeros((0, context), dtype=np.float32), np.zeros((0, horizon), dtype=np.float32)

    starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
    X = np.stack([series[s:s+context] for s in starts], axis=0).astype(np.float32)
    Y = np.stack([series[s+context:s+context+horizon] for s in starts], axis=0).astype(np.float32)
    return X, Y


def find_target_linears(model: nn.Module, include_regex: str, exclude_regex: str) -> List[Tuple[str, nn.Linear]]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    out: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if inc is not None and inc.match(name) is None:
                continue
            if exc is not None and exc.match(name) is not None:
                continue
            out.append((name, mod))
    return out


# -------------------------
# Error weights -> signal/noise weights
# -------------------------

def compute_window_errors(pipe, pipe_kind: str, X_pool: np.ndarray, Y_pool: np.ndarray, horizon: int, batch: int, num_samples: int) -> np.ndarray:
    """Per-window MSE on calibration pool."""
    n = X_pool.shape[0]
    errs = np.zeros((n,), dtype=np.float64)
    for i in range(0, n, batch):
        xb = X_pool[i:i+batch]
        yb = Y_pool[i:i+batch]
        pb = forecast_batch(pipe, pipe_kind, xb, horizon, batch_size=batch, num_samples=num_samples)
        d = pb - yb
        errs[i:i+batch] = np.mean(d * d, axis=1)
    return errs


def normalize_clip(w: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    m = float(np.mean(w)) if w.size > 0 else 1.0
    if m <= 0:
        m = 1.0
    w = w / m
    w = np.clip(w, clip_min, clip_max)
    # renormalize to mean 1 after clipping (helps stability)
    m2 = float(np.mean(w)) if w.size > 0 else 1.0
    if m2 <= 0:
        m2 = 1.0
    return (w / m2).astype(np.float32)


# -------------------------
# Signal/Noise Gram collection (FIXED)
# -------------------------

@dataclass
class GramStat:
    # Weighted grams for SNR expert
    Gsig: torch.Tensor          # [G,4,4] CPU float32
    Csig: float
    Gnoi: Optional[torch.Tensor]
    Cnoi: float

    # Unweighted activation gram (for WANDA expert) + gate stats
    Gact: torch.Tensor          # [G,4,4] CPU float32
    Cact: float
    m2: float                   # sum(x^2) over sampled activations
    m4: float                   # sum(x^4) over sampled activations
    n: int                      # count of activation elements used for moments
    dx2: float                  # sum((x_t - x_{t-1})^2) (if x is [B,T,C])
    ndx: int


def expand_window_weights_to_B(w: torch.Tensor, B: int) -> torch.Tensor:
    """Repeat weights to length B (best-effort alignment when model flattens tokens/patches)."""
    w = w.flatten()
    if w.numel() == B:
        return w
    if w.numel() == 0:
        return w.new_ones((B,))
    reps = int(np.ceil(B / w.numel()))
    out = w.repeat(reps)[:B]
    return out


def softmax3(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def compute_season_trend_strength(X_sel: np.ndarray, ma_win: int, eps: float = 1e-8) -> Tuple[float, float, int]:
    """Cheap season/trend strength proxy on the *input* windows.

    Returns: (season_strength, trend_strength, used_ma_win)
    Strengths are ratios in [0, ~1] computed from variance decomposition:
      trend = moving average
      season = x - trend
      strength = var(component) / var(x)
    """
    X = np.asarray(X_sel, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    L = int(X.shape[1])
    if L < 8:
        return 0.5, 0.5, 1

    W = int(ma_win)
    W = max(3, min(W, L - 1))
    if W % 2 == 0:
        W += 1
    pad = W // 2

    # reflect padding + cumsum moving average (vectorized)
    Xpad = np.pad(X, ((0, 0), (pad, pad)), mode="reflect")
    c = np.cumsum(Xpad, axis=1, dtype=np.float64)
    ma = (c[:, W:] - c[:, :-W]) / float(W)  # [N,L]

    season = X.astype(np.float64) - ma
    varx = np.var(X.astype(np.float64), axis=1) + eps
    trend_strength = np.var(ma, axis=1) / varx
    season_strength = np.var(season, axis=1) / varx

    return float(np.mean(season_strength)), float(np.mean(trend_strength)), W

def gate_for_layer(st: GramStat,
                   season_strength: float,
                   trend_strength: float,
                   eps: float = 1e-8,
                   temp: float = 1.0,
                   floor: float = 0.05) -> Tuple[Tuple[float, float, float], float, float]:
    """Return (gate_weights, excess_kurtosis, roughness)."""
    n = max(int(st.n), 1)
    v = (st.m2 / n) + eps
    k_excess = (st.m4 / n) / (v * v) - 3.0

    # roughness in units of variance; only meaningful when we saw [B,T,C]
    if int(st.ndx) > 0:
        r = (st.dx2 / max(int(st.ndx), 1)) / v
    else:
        r = 0.0

    relu_k = max(0.0, k_excess)

    # logits: push SNR when season/trend strong but activations are not too heavy-tailed
    logit_snr   = +1.2 * season_strength + 0.6 * trend_strength - 0.8 * relu_k - 0.4 * r
    logit_wanda = +1.0 * relu_k + 0.3 * r + 0.3 * season_strength
    logit_mag   = 0.0

    w = softmax3([logit_snr / max(temp, 1e-6),
                  logit_wanda / max(temp, 1e-6),
                  logit_mag / max(temp, 1e-6)])
    w = np.maximum(w, float(floor))
    w = w / np.sum(w)

    gate = (float(w[0]), float(w[1]), float(w[2]))
    return gate, float(k_excess), float(r)



@torch.no_grad()
def collect_group_grams_signal_noise(
    pipe,
    pipe_kind: str,
    targets: List[Tuple[str, nn.Linear]],
    X_sel: np.ndarray,
    w_sig_sel: np.ndarray,            # [K]
    w_noi_sel: Optional[np.ndarray],  # [K] or None
    horizon: int,
    calib_batch: int,
    sample_rows_per_call: int,
    max_calls_per_layer: int,
    num_samples: int,
) -> Dict[str, GramStat]:
    stats: Dict[str, GramStat] = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []

    # current per-window weights for this outer batch
    current_wsig: Optional[torch.Tensor] = None
    current_wnoi: Optional[torch.Tensor] = None
    warned_token = False

    def make_hook(layer_name: str):
        def pre_hook(_mod, inputs):
            nonlocal warned_token
            if calls[layer_name] >= max_calls_per_layer:
                return

            x = inputs[0]
            if not torch.is_floating_point(x):
                return

            B = int(x.shape[0])
            if current_wsig is None:
                raise RuntimeError("Internal error: current_wsig not set.")

            # start from per-window weights (len == outer calib_batch)
            ws_batch = current_wsig.to(x.device).float()
            wn_batch = None if current_wnoi is None else current_wnoi.to(x.device).float()

            # Chronos-2 often flattens tokens/patches so hook B != calib_batch.
            if ws_batch.numel() != B:
                if not warned_token:
                    print(f"[warn] internal batch mismatch: weights={ws_batch.numel()} but hook sees B={B}. "
                          f"Treating B as token/patch batch and repeating per-window weights.")
                    warned_token = True
                ws_batch = expand_window_weights_to_B(ws_batch, B)
                if wn_batch is not None:
                    wn_batch = expand_window_weights_to_B(wn_batch, B)

            # flatten x -> [N, C]
            if x.dim() == 3:
                xf = x.reshape(-1, x.shape[-1])
                # expand weights across time dim (approx)
                T = x.shape[1]
                ws_exp = ws_batch.unsqueeze(1).expand(B, T).reshape(-1)
                wn_exp = None if wn_batch is None else wn_batch.unsqueeze(1).expand(B, T).reshape(-1)
            else:
                xf = x.reshape(x.shape[0], -1)
                ws_exp = ws_batch
                wn_exp = wn_batch

            xf = xf.float()
            ws_exp = ws_exp.float()
            if wn_exp is not None:
                wn_exp = wn_exp.float()

            # group into 4-wide groups along channel dim
            Cin = int(xf.shape[-1])
            G = Cin // 4
            Cg = G * 4
            if Cg == 0:
                return
            xf = xf[:, :Cg]

            # subsample rows
            if xf.shape[0] > sample_rows_per_call:
                idx = torch.randint(0, xf.shape[0], (sample_rows_per_call,), device=xf.device)
                xf = xf.index_select(0, idx)
                ws_exp = ws_exp.index_select(0, idx)
                if wn_exp is not None:
                    wn_exp = wn_exp.index_select(0, idx)

            N = int(xf.shape[0])
            xg = xf.reshape(N, G, 4)  # [N,G,4]

            # --- Unweighted gram for WANDA expert ---
            wu = torch.ones_like(ws_exp)
            Gact_batch = torch.einsum("n,ngc,ngd->gcd", wu, xg, xg).detach().cpu()
            Cact = float(wu.sum().item())

            # --- Activation moments for gating (kurtosis proxy) ---
            vals = xf  # [N, Cg] float32
            m2 = float((vals * vals).sum().item())
            m4 = float(((vals * vals) ** 2).sum().item())
            n_m = int(vals.numel())

            # --- Temporal roughness (only if x is [B,T,C]) ---
            dx2 = 0.0
            ndx = 0
            if x.dim() == 3 and x.shape[1] > 1 and Cg > 0:
                x3 = x[:, :, :Cg].float()
                dx = x3[:, 1:, :] - x3[:, :-1, :]
                flat = dx.reshape(-1)
                if flat.numel() > sample_rows_per_call * 4:
                    ridx = torch.randint(0, flat.numel(), (sample_rows_per_call * 4,), device=flat.device)
                    flat = flat.index_select(0, ridx)
                dx2 = float((flat * flat).sum().item())
                ndx = int(flat.numel())

            # --- Weighted grams for SNR expert ---
            Gs_batch = torch.einsum("n,ngc,ngd->gcd", ws_exp, xg, xg).detach().cpu()
            Cs = float(ws_exp.sum().item())

            Gn_batch = None
            Cn = 0.0
            if wn_exp is not None:
                Gn_batch = torch.einsum("n,ngc,ngd->gcd", wn_exp, xg, xg).detach().cpu()
                Cn = float(wn_exp.sum().item())

            if layer_name not in stats:
                stats[layer_name] = GramStat(
                    Gsig=Gs_batch, Csig=Cs, Gnoi=Gn_batch, Cnoi=Cn,
                    Gact=Gact_batch, Cact=Cact,
                    m2=m2, m4=m4, n=n_m,
                    dx2=dx2, ndx=ndx,
                )
            else:
                st = stats[layer_name]
                st.Gsig += Gs_batch
                st.Csig += Cs
                if Gn_batch is not None:
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

            calls[layer_name] += 1
        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    n = int(X_sel.shape[0])
    for i in range(0, n, calib_batch):
        end = min(i + calib_batch, n)
        current_wsig = torch.from_numpy(w_sig_sel[i:end]).float()
        current_wnoi = None if w_noi_sel is None else torch.from_numpy(w_noi_sel[i:end]).float()

        xb = X_sel[i:end]
        _ = forecast_batch(pipe, pipe_kind, xb, horizon, batch_size=calib_batch, num_samples=num_samples)

    for h in hooks:
        h.remove()
    return stats


# -------------------------
# 2:4 pruning using SNR grams
# -------------------------

@torch.no_grad()
def prune_linear_snr_2of4(layer: nn.Linear, Gsig: torch.Tensor, Gnoi: Optional[torch.Tensor], score_mode: str, eps: float) -> None:
    """
    Computes a per-weight score from diag(Gsig)/diag(Gnoi) and selects top-2 in each 4-group.

    score_mode:
      - ratio:  |w| * sqrt( (ds + eps) / (dn + eps) )
      - keep :  do nothing (useful for debugging)
    """
    if score_mode == "keep":
        return

    W = layer.weight.data
    Cin = int(W.shape[1])
    G = Cin // 4
    Cg = G * 4
    if Cg == 0:
        return

    # diag stats [G,4]
    ds = torch.diagonal(Gsig, dim1=-2, dim2=-1).float()  # CPU float
    if Gnoi is None:
        dn = torch.ones_like(ds)
    else:
        dn = torch.diagonal(Gnoi, dim1=-2, dim2=-1).float()

    # stabilize
    ds = ds.clamp_min(0.0) + eps
    dn = dn.clamp_min(0.0) + eps

    # move to device
    ds = ds.to(W.device, dtype=torch.float32)
    dn = dn.to(W.device, dtype=torch.float32)

    scale = torch.sqrt(ds / dn)  # [G,4]
    Wf = W[:, :Cg].float().view(W.shape[0], G, 4)  # [O,G,4]

    scores = Wf.abs() * scale.unsqueeze(0)
    top2 = torch.topk(scores, k=2, dim=2, largest=True).indices
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask.scatter_(2, top2, True)

    Wf = Wf * mask.to(Wf.dtype)
    W[:, :Cg] = Wf.reshape(W.shape[0], Cg).to(W.dtype)
    if Cg < Cin:
        W[:, Cg:] = 0


# -------------------------

@torch.no_grad()
def prune_linear_fused_2of4(
    layer: nn.Linear,
    Gsig: torch.Tensor,
    Gnoi: Optional[torch.Tensor],
    Gact: torch.Tensor,
    gate: Tuple[float, float, float],
    eps: float,
) -> None:
    """Soft fusion of experts (SNR + WANDA + MAG), then strict top-2 per 4-group."""
    a, b, c = gate  # (snr, wanda, mag)

    W = layer.weight.data
    Cin = int(W.shape[1])
    G = Cin // 4
    Cg = G * 4
    if Cg == 0:
        return

    # diag stats [G,4] on CPU -> float32 on device
    ds = torch.diagonal(Gsig, dim1=-2, dim2=-1).float()
    dn = torch.ones_like(ds) if Gnoi is None else torch.diagonal(Gnoi, dim1=-2, dim2=-1).float()
    da = torch.diagonal(Gact, dim1=-2, dim2=-1).float()

    ds = ds.clamp_min(0.0) + eps
    dn = dn.clamp_min(0.0) + eps
    da = da.clamp_min(0.0) + eps

    ds = ds.to(W.device, dtype=torch.float32)
    dn = dn.to(W.device, dtype=torch.float32)
    da = da.to(W.device, dtype=torch.float32)

    scale_snr = torch.sqrt(ds / dn)     # [G,4]
    scale_wan = torch.sqrt(da)          # [G,4]

    Wf = W[:, :Cg].float().view(W.shape[0], G, 4)  # [O,G,4]

    s_mag = Wf.abs()
    s_wan = Wf.abs() * scale_wan.unsqueeze(0)
    s_snr = Wf.abs() * scale_snr.unsqueeze(0)

    # per-group normalize each expert so one can't dominate due to scale
    def norm4(s):
        denom = (s.mean(dim=2, keepdim=True) + 1e-8)
        return s / denom

    s_mag = norm4(s_mag)
    s_wan = norm4(s_wan)
    s_snr = norm4(s_snr)

    scores = a * s_snr + b * s_wan + c * s_mag

    top2 = torch.topk(scores, k=2, dim=2, largest=True).indices
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask.scatter_(2, top2, True)

    Wf = Wf * mask.to(Wf.dtype)
    W[:, :Cg] = Wf.reshape(W.shape[0], Cg).to(W.dtype)
    if Cg < Cin:
        W[:, Cg:] = 0


# Args / main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cols_regex", default="^OT$")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--train_end", type=int, default=None)

    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_test", type=int, default=96)

    ap.add_argument("--calib_windows", type=int, default=1024)
    ap.add_argument("--calib_select", choices=["first", "random", "topk"], default="first")
    ap.add_argument("--calib_batch", type=int, default=1)
    ap.add_argument("--max_calls_per_layer", type=int, default=64)
    ap.add_argument("--sample_rows_per_call", type=int, default=2048)

    ap.add_argument("--test_windows", type=int, default=-1)  # -1 => all
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--measure_time", action="store_true")

    ap.add_argument("--model_id", default="amazon/chronos-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch_dtype", choices=["None", "float16", "bfloat16", "float32"], default="None")
    ap.add_argument("--num_samples", type=int, default=16)

    ap.add_argument("--include_regex", default=".*")
    ap.add_argument("--exclude_regex", default=r".*(embed|embedding|patch|token|input|pos|lm_head|head|output).*")

    ap.add_argument("--error_power", type=float, default=1.0)
    ap.add_argument("--sn_gamma", type=float, default=1.0)
    ap.add_argument("--score_mode", choices=["ratio", "fused", "keep"], default="ratio")
    ap.add_argument("--fusion_temp", type=float, default=1.0, help="Softmax temperature for fused gating (lower => sharper).")
    ap.add_argument("--fusion_floor", type=float, default=0.05, help="Minimum probability per expert in fused gating.")
    ap.add_argument("--print_gate", action="store_true", help="Print per-layer fusion weights and stats.")
    ap.add_argument("--decomp_ma", type=int, default=49, help="Moving-average window (on input) for season/trend proxy.")
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--w_clip_min", type=float, default=0.25)
    ap.add_argument("--w_clip_max", type=float, default=4.0)

    ap.add_argument("--seed", type=int, default=2026)

    # accept --zscore like your other script, default ON
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--zscore", dest="zscore", action="store_true")
    g.add_argument("--no_zscore", dest="zscore", action="store_false")
    ap.set_defaults(zscore=True)

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    cols = [c for c in df.columns if re.match(args.cols_regex, c)]
    if not cols:
        raise ValueError(f"No columns match cols_regex={args.cols_regex}")
    data = df[cols].to_numpy(dtype=np.float32)
    print(f"[data] rows={len(df)} cols={len(cols)} cols_regex={args.cols_regex} -> {cols}")

    T = data.shape[0]
    train_end = args.train_end if args.train_end is not None else int(0.7 * T)
    train_end = min(max(int(train_end), 0), T)
    if train_end < args.context + args.horizon + 1:
        raise ValueError(f"train_end too small ({train_end}) for context+horizon={args.context+args.horizon}")

    mu = data[:train_end].mean(axis=0)
    sigma = data[:train_end].std(axis=0) + 1e-12

    if args.zscore:
        data_n = (data - mu) / sigma
    else:
        data_n = data.copy()
        sigma = np.ones_like(sigma)

    mse_base_list, mae_base_list = [], []
    mse_pr_list, mae_pr_list = [], []
    mse_base_raw_list, mae_base_raw_list = [], []
    mse_pr_raw_list, mae_pr_raw_list = [], []

    for ci, col in enumerate(cols):
        # IMPORTANT: reload model per column so pruning doesn't carry over
        pipe, pipe_kind = load_chronos(args.model_id, args.device, args.torch_dtype)
        model = getattr(pipe, "model", pipe)

        targets = find_target_linears(model, args.include_regex, args.exclude_regex)
        if ci == 0:
            print(f"[model] {args.model_id} | pipe_kind={pipe_kind} | targets(nn.Linear)={len(targets)}")

        series_train = data_n[:train_end, ci]
        series_test = data_n[train_end:, ci]

        X_train, Y_train = build_windows(series_train, args.context, args.horizon, args.stride_train)
        X_test, Y_test = build_windows(series_test, args.context, args.horizon, args.stride_test)

        if args.test_windows != -1:
            X_test = X_test[:args.test_windows]
            Y_test = Y_test[:args.test_windows]

        poolN = min(args.calib_windows, X_train.shape[0]) if args.calib_windows != -1 else X_train.shape[0]
        if args.calib_select == "first":
            pool_idx = np.arange(poolN, dtype=np.int64)
        else:
            pool_idx = np.random.permutation(X_train.shape[0]).astype(np.int64)[:poolN]

        X_pool = X_train[pool_idx]
        Y_pool = Y_train[pool_idx]

        # Budget for grams collection: K windows, not necessarily full pool
        K = min(poolN, args.max_calls_per_layer * args.calib_batch)

        print(f"\n[col={col}] train_windows={X_train.shape[0]} calib_pool={poolN} test_windows={X_test.shape[0]} K={K}")

        # baseline
        if args.measure_time:
            base, avg_batch_sec = timed_eval(pipe, pipe_kind, X_test, Y_test, args.horizon, args.batch, args.num_samples)
        else:
            base = eval_model(pipe, pipe_kind, X_test, Y_test, args.horizon, args.batch, args.num_samples)
            avg_batch_sec = float("nan")

        mse_b, mae_b = base["mse"], base["mae"]
        sig = float(sigma[ci])
        mse_b_raw = mse_b * (sig ** 2)
        mae_b_raw = mae_b * sig

        print(f"[baseline] MSE_norm={mse_b:.6f} MAE_norm={mae_b:.6f} | "
              f"MSE_raw={mse_b_raw:.6f} MAE_raw={mae_b_raw:.6f} | sigma={sig:.4f}")
        if args.measure_time:
            print(f"[baseline] avg_batch_sec={avg_batch_sec:.4f}")

        # error weights on pool (label-assisted calibration, train-only)
        print(f"[calib] computing errors/weights on pool: power={args.error_power} ...")
        errs = compute_window_errors(pipe, pipe_kind, X_pool, Y_pool, args.horizon, batch=max(1, args.batch), num_samples=args.num_samples)

        w = (errs + args.eps) ** float(args.error_power)
        w_sig = normalize_clip(w, args.w_clip_min, args.w_clip_max)

        # noise emphasizes "easy" windows
        w_inv = (1.0 / (errs + args.eps)) ** float(args.sn_gamma)
        w_noi = normalize_clip(w_inv, args.w_clip_min, args.w_clip_max)

        print(f"[calib] weight_stats(sig): min={w_sig.min():.4f} max={w_sig.max():.4f} mean={w_sig.mean():.4f}")
        print(f"[calib] weight_stats(noi): min={w_noi.min():.4f} max={w_noi.max():.4f} mean={w_noi.mean():.4f}")

        # select K windows for grams
        if args.calib_select == "topk":
            sel_idx = np.argsort(-w_sig)[:K]
        elif args.calib_select == "random":
            sel_idx = np.random.permutation(poolN)[:K]
        else:
            sel_idx = np.arange(K, dtype=np.int64)

        X_sel = X_pool[sel_idx]
        w_sig_sel = w_sig[sel_idx]
        w_noi_sel = w_noi[sel_idx]

        print(f"[calib] collecting grams: select={args.calib_select} pool={poolN} K={K} "
              f"(max_calls_per_layer={args.max_calls_per_layer}, calib_batch={args.calib_batch})")

        grams = collect_group_grams_signal_noise(
            pipe=pipe,
            pipe_kind=pipe_kind,
            targets=targets,
            X_sel=X_sel,
            w_sig_sel=w_sig_sel,
            w_noi_sel=w_noi_sel,
            horizon=args.horizon,
            calib_batch=args.calib_batch,
            sample_rows_per_call=args.sample_rows_per_call,
            max_calls_per_layer=args.max_calls_per_layer,
            num_samples=args.num_samples,
        )
        print(f"[calib] collected grams for {len(grams)}/{len(targets)} layers")

        # prune
        if args.score_mode != "keep":
            if args.score_mode == "fused":
                S, T, usedW = compute_season_trend_strength(X_sel, args.decomp_ma, eps=args.eps)
                print(f"[gate] input season_strength={S:.3f} trend_strength={T:.3f} (ma={usedW})")
                print(f"[prune] FUSED 2:4: temp={args.fusion_temp:g} floor={args.fusion_floor:g}")
                for name, layer in targets:
                    st = grams.get(name, None)
                    if st is None:
                        continue
                    Gs = st.Gsig / max(st.Csig, args.eps)
                    Gn = None if st.Gnoi is None else (st.Gnoi / max(st.Cnoi, args.eps))
                    Ga = st.Gact / max(st.Cact, args.eps)
                    gate, k_ex, r = gate_for_layer(st, S, T, eps=args.eps, temp=args.fusion_temp, floor=args.fusion_floor)
                    prune_linear_fused_2of4(layer, Gs, Gn, Ga, gate, eps=args.eps)
                    if args.print_gate:
                        print(f"[gate] {name}: snr={gate[0]:.2f} wanda={gate[1]:.2f} mag={gate[2]:.2f} | kurt={k_ex:.2f} rough={r:.2f}")
            else:
                print(f"[prune] SNR 2:4: score_mode={args.score_mode}")
                for name, layer in targets:
                    st = grams.get(name, None)
                    if st is None:
                        continue
                    # normalize grams
                    Gs = st.Gsig / max(st.Csig, args.eps)
                    Gn = None if st.Gnoi is None else (st.Gnoi / max(st.Cnoi, args.eps))
                    prune_linear_snr_2of4(layer, Gs, Gn, score_mode=args.score_mode, eps=args.eps)
        # eval pruned
        pr = eval_model(pipe, pipe_kind, X_test, Y_test, args.horizon, args.batch, args.num_samples)
        mse_p, mae_p = pr["mse"], pr["mae"]
        mse_p_raw = mse_p * (sig ** 2)
        mae_p_raw = mae_p * sig

        print(f"[pruned]   MSE_norm={mse_p:.6f} MAE_norm={mae_p:.6f} | "
              f"MSE_raw={mse_p_raw:.6f} MAE_raw={mae_p_raw:.6f}")
        print(f"[delta]   ΔMSE_norm={mse_p - mse_b:+.6f} ΔMAE_norm={mae_p - mae_b:+.6f} | "
              f"ΔMSE_raw={mse_p_raw - mse_b_raw:+.6f} ΔMAE_raw={mae_p_raw - mae_b_raw:+.6f}")

        mse_base_list.append(mse_b); mae_base_list.append(mae_b)
        mse_pr_list.append(mse_p);   mae_pr_list.append(mae_p)
        mse_base_raw_list.append(mse_b_raw); mae_base_raw_list.append(mae_b_raw)
        mse_pr_raw_list.append(mse_p_raw);   mae_pr_raw_list.append(mae_p_raw)

    print("\n==============================")
    print("FINAL (avg across columns)")
    print("==============================")
    print(f"Baseline (norm): MSE={float(np.mean(mse_base_list)):.6f} MAE={float(np.mean(mae_base_list)):.6f}")
    print(f"Pruned   (norm): MSE={float(np.mean(mse_pr_list)):.6f} MAE={float(np.mean(mae_pr_list)):.6f}")
    print(f"Delta    (norm): ΔMSE={float(np.mean(np.array(mse_pr_list)-np.array(mse_base_list))):+.6f} "
          f"ΔMAE={float(np.mean(np.array(mae_pr_list)-np.array(mae_base_list))):+.6f}")
    print(f"Baseline (raw):  MSE={float(np.mean(mse_base_raw_list)):.6f} MAE={float(np.mean(mae_base_raw_list)):.6f}")
    print(f"Pruned   (raw):  MSE={float(np.mean(mse_pr_raw_list)):.6f} MAE={float(np.mean(mae_pr_raw_list)):.6f}")
    print(f"Delta    (raw):  ΔMSE={float(np.mean(np.array(mse_pr_raw_list)-np.array(mse_base_raw_list))):+.6f} "
          f"ΔMAE={float(np.mean(np.array(mae_pr_raw_list)-np.array(mae_base_raw_list))):+.6f}")


if __name__ == "__main__":
    main()

