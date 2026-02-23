#!/usr/bin/env python3
"""
snr_2of4_error_weighted_fair_multicol.py

Multi-column (many univariate series) variant of snr_2of4_error_weighted_fair_v3.py.

Key changes:
- Supports --cols_regex or --cols (comma list). Computes macro-average over columns.
- Memory-safe: does NOT build all windows for each series (Electricity has 320 cols).
- Calibration pool is built by sampling windows across columns (balanced round-robin for 'first').
- Prints both macro (avg over columns) and micro (global) MSE/MAE.

Fairness:
- Error weighting uses ONLY train/calib labels (Y_pool). Never uses test labels.
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
    # TimesFM API expects list of arrays (each is context-length)
    inputs = [X[i].astype(np.float32) for i in range(X.shape[0])]
    point_forecast, _quant = tfm_model.forecast(horizon=horizon, inputs=inputs)
    return np.asarray(point_forecast, dtype=np.float32)

# -------------------------
# Data: multi-series
# -------------------------
def load_multiseries(csv_path: str,
                     cols_regex: Optional[str],
                     cols_list: Optional[str],
                     drop_cols_regex: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)

    if cols_list:
        cols = [c.strip() for c in cols_list.split(",") if c.strip()]
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Column {c} not found in {csv_path}.")
    elif cols_regex:
        cre = re.compile(cols_regex)
        cols = [c for c in df.columns if cre.match(c)]
        if not cols:
            raise ValueError(f"No columns match cols_regex={cols_regex}")
    else:
        raise ValueError("Provide either --cols_regex or --cols (comma-separated).")

    if drop_cols_regex:
        dre = re.compile(drop_cols_regex)
        cols = [c for c in cols if not dre.match(c)]
        if not cols:
            raise ValueError("After drop_cols_regex, no columns remain.")

    return df, cols

def _window_from_series(series: np.ndarray, start: int, context: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    x = series[start:start+context]
    y = series[start+context:start+context+horizon]
    return x, y

def _count_windows(start: int, end: int, context: int, horizon: int, stride: int) -> int:
    # window start positions: start .. last inclusive by stride
    last = end - (context + horizon)
    if last < start:
        return 0
    return ((last - start) // stride) + 1

def _iter_window_batches(series: np.ndarray,
                         start: int,
                         end: int,
                         context: int,
                         horizon: int,
                         stride: int,
                         start_window_idx: int,
                         max_windows: int,
                         batch: int):
    """
    Yields (Xb, Yb) batches where Xb shape [B, context], Yb shape [B, horizon].
    start_window_idx is an index in the window sequence (not time index).
    """
    last = end - (context + horizon)
    if last < start:
        return

    # compute first window start position respecting start_window_idx
    first_i = start + start_window_idx * stride
    if first_i > last:
        return

    # how many windows available from first_i
    n_avail = ((last - first_i) // stride) + 1
    if max_windows < 0:
        n_use = n_avail
    else:
        n_use = min(n_avail, max_windows)

    # iterate
    count = 0
    i = first_i
    while count < n_use:
        curB = min(batch, n_use - count)
        Xb = np.empty((curB, context), dtype=np.float32)
        Yb = np.empty((curB, horizon), dtype=np.float32)
        for j in range(curB):
            x, y = _window_from_series(series, i, context, horizon)
            Xb[j] = x
            Yb[j] = y
            i += stride
        yield Xb, Yb
        count += curB

# -------------------------
# Metrics
# -------------------------
@dataclass
class Agg:
    sse: float = 0.0
    sae: float = 0.0
    n: int = 0  # number of scalar points (windows*horizon)

def mse_mae_from_agg(a: Agg) -> Tuple[float, float]:
    if a.n <= 0:
        return float("nan"), float("nan")
    return a.sse / a.n, a.sae / a.n

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
# Calibration pool builder (balanced across columns)
# -------------------------
def build_calib_pool(df: pd.DataFrame,
                     cols: List[str],
                     pool_size: int,
                     train_end: int,
                     context: int,
                     horizon: int,
                     stride_train: int,
                     seed: int,
                     mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_pool: [P, context]
      Y_pool: [P, horizon]
      sid:    [P] series index (0..S-1) for each window
    """
    rng = np.random.default_rng(seed)
    S = len(cols)
    data = df[cols].to_numpy(dtype=np.float32)  # [T,S]
    T = data.shape[0]

    # total possible starts per series in train region [0, train_end)
    last = train_end - (context + horizon)
    if last < 0:
        raise ValueError("train_end too small for context+horizon.")
    n_starts = (last // stride_train) + 1

    P = int(pool_size)
    Xp = np.empty((P, context), dtype=np.float32)
    Yp = np.empty((P, horizon), dtype=np.float32)
    sid = np.empty((P,), dtype=np.int64)

    if mode == "first":
        # balanced round-robin over (window_idx, series)
        k = 0
        w_idx = 0
        while k < P:
            any_added = False
            i = w_idx * stride_train
            if i > last:
                break
            for s in range(S):
                if k >= P:
                    break
                series = data[:, s]
                x, y = _window_from_series(series, i, context, horizon)
                Xp[k] = x
                Yp[k] = y
                sid[k] = s
                k += 1
                any_added = True
            if not any_added:
                break
            w_idx += 1
        if k < P:
            Xp = Xp[:k]
            Yp = Yp[:k]
            sid = sid[:k]

    elif mode == "random":
        # sample (series, start_idx) uniformly with replacement
        # start_idx here is the window index in [0, n_starts)
        s_idx = rng.integers(low=0, high=S, size=P, endpoint=False)
        w_idx = rng.integers(low=0, high=n_starts, size=P, endpoint=False)
        for k in range(P):
            s = int(s_idx[k])
            i = int(w_idx[k]) * stride_train
            series = data[:, s]
            x, y = _window_from_series(series, i, context, horizon)
            Xp[k] = x
            Yp[k] = y
            sid[k] = s

    else:
        raise ValueError(f"Unknown calib_pool_mode: {mode}")

    return Xp, Yp, sid

# -------------------------
# Gram Collection (grouped 4)
# -------------------------
@dataclass
class GramStat:
    Gsum: torch.Tensor
    count: float

@torch.no_grad()
def compute_errors_and_weights(
    tfm_model,
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    sid_pool: np.ndarray,
    horizon: int,
    calib_batch: int,
    error_power: float,
    weight_norm: str,
):
    """
    Returns:
      errors: [N] MSE per window
      weights: [N] normalized weights
    """
    preds = []
    for i in range(0, len(X_pool), calib_batch):
        preds.append(forecast_timesfm_point(tfm_model, X_pool[i:i+calib_batch], horizon))
    preds = np.concatenate(preds, axis=0)  # [N, H]
    diff = preds - Y_pool
    errors = np.mean(diff**2, axis=1).astype(np.float32)  # [N]

    if weight_norm == "global":
        mean_err = float(np.mean(errors))
        weights = (errors / (mean_err + 1e-6)) ** float(error_power)
    elif weight_norm == "per_series":
        # normalize error by mean error within each series (macro-fair)
        weights = np.empty_like(errors)
        for s in np.unique(sid_pool):
            m = (sid_pool == s)
            mean_s = float(np.mean(errors[m])) if np.any(m) else float(np.mean(errors))
            weights[m] = (errors[m] / (mean_s + 1e-6)) ** float(error_power)
        # re-normalize so mean weight ~ 1 globally (stability)
        weights = weights / (float(np.mean(weights)) + 1e-6)
    else:
        raise ValueError("weight_norm must be global or per_series")

    weights = weights.astype(np.float32)
    return errors, weights

@torch.no_grad()
def collect_group_grams_weighted(
    tfm_model,
    targets: List[Tuple[str, nn.Linear]],
    X_sel: np.ndarray,
    weights_sel: np.ndarray,          # [K]
    horizon: int,
    calib_batch: int,
    sample_rows_per_call: int,
    max_calls_per_layer: int,
):
    stats: Dict[str, GramStat] = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []

    global current_batch_weights
    current_batch_weights = None
    warned_split = False

    def make_hook(layer_name: str):
        def pre_hook(_mod, inputs):
            nonlocal warned_split
            if calls[layer_name] >= max_calls_per_layer:
                return
            (x,) = inputs
            B = x.shape[0]
            if current_batch_weights is None:
                raise RuntimeError("Internal error: current_batch_weights not set.")
            w_batch = current_batch_weights.to(x.device)

            if w_batch.numel() != B:
                if not warned_split:
                    print(f"[warn] calib batch mismatch inside model: weights={w_batch.numel()} but hook sees B={B}. "
                          f"TimesFM likely micro-batched. Set --calib_batch smaller (e.g., 4 or 1).")
                    warned_split = True
                if w_batch.numel() >= B:
                    w_batch = w_batch[:B]
                else:
                    pad = w_batch.new_full((B - w_batch.numel(),), float(w_batch[-1].item()))
                    w_batch = torch.cat([w_batch, pad], dim=0)

            if x.dim() == 3:
                T = x.shape[1]
                xf = x.reshape(-1, x.shape[-1])
                w_expanded = w_batch.unsqueeze(1).expand(B, T).reshape(-1)
            else:
                xf = x
                w_expanded = w_batch

            G = xf.shape[-1] // 4
            Cg = G * 4
            if Cg == 0:
                return
            xf = xf[:, :Cg]

            if xf.shape[0] > sample_rows_per_call:
                idx = torch.randint(0, xf.shape[0], (sample_rows_per_call,), device=xf.device)
                xf = xf.index_select(0, idx)
                w_expanded = w_expanded.index_select(0, idx)

            xg = xf.reshape(xf.shape[0], G, 4)
            G_batch = torch.einsum("n,ngc,ngd->gcd", w_expanded, xg, xg).cpu()
            w_sum = float(w_expanded.sum().item())

            if layer_name not in stats:
                stats[layer_name] = GramStat(Gsum=G_batch, count=w_sum)
            else:
                stats[layer_name].Gsum += G_batch
                stats[layer_name].count += w_sum

            calls[layer_name] += 1
        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    n = X_sel.shape[0]
    for i in range(0, n, calib_batch):
        end = min(i + calib_batch, n)
        current_batch_weights = torch.from_numpy(weights_sel[i:end]).float()
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
    i,j = PAIRS[k].tolist()
    PAIR_MASKS[k,i] = PAIR_MASKS[k,j] = 1.0

@torch.no_grad()
def prune_linear_snr_2of4(layer, Gsum, count, score_mode, eps, refit, ridge):
    W = layer.weight.data
    O, C = W.shape
    Ggroups = C // 4
    Cg = Ggroups * 4
    if Cg == 0:
        return
    device = W.device
    dtype = W.dtype

    Wg = W[:, :Cg].view(O, Ggroups, 4)
    Gg = (Gsum / max(count, 1e-6)).to(device=device, dtype=dtype)
    masks = PAIR_MASKS.to(device=device, dtype=dtype)
    scores = torch.empty((O, Ggroups, 6), device=device, dtype=dtype)

    for k in range(6):
        mk = masks[k].view(1,1,4)
        md = (1.0 - mk)
        Wk = Wg * mk
        Wd = Wg * md
        Tk = torch.einsum("ogc,gcd->ogd", Wk, Gg)
        Ek = (Tk * Wk).sum(dim=2)
        Td = torch.einsum("ogc,gcd->ogd", Wd, Gg)
        Ed = (Td * Wd).sum(dim=2)
        scores[:, :, k] = Ek if score_mode == "keep" else Ek / (Ed + eps)

    bestk = torch.argmax(scores, dim=2)
    pair_idx = PAIRS.to(device=device)[bestk]
    keep_mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool)
    keep_mask.scatter_(2, pair_idx, torch.ones_like(pair_idx, dtype=torch.bool))

    if not refit:
        Wnew = torch.where(keep_mask, Wg, torch.zeros_like(Wg))
    else:
        B = torch.einsum("ogc,gcd->ogd", Wg, Gg)
        invs = []
        eye2 = torch.eye(2, device=device, dtype=dtype).view(1,2,2)
        for k in range(6):
            i, j = PAIRS[k].tolist()
            Gss = Gg[:, [i,j]][:, :, [i,j]] + ridge * eye2
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

# -------------------------
# Evaluation over all columns (macro + micro)
# -------------------------
def eval_all_series(tfm,
                    df: pd.DataFrame,
                    cols: List[str],
                    start: int,
                    end: int,
                    context: int,
                    horizon: int,
                    stride: int,
                    test_start_window: int,
                    test_windows: int,
                    batch: int,
                    measure_time: bool):
    data = df[cols].to_numpy(dtype=np.float32)  # [T,S]
    S = data.shape[1]

    micro = Agg()
    per_series_mse = []
    per_series_mae = []
    times = []

    for s in range(S):
        series = data[:, s]
        agg = Agg()
        # warmup a couple batches per series (optional)
        # (kept minimal to not explode runtime)
        warmed = 0
        for Xb, Yb in _iter_window_batches(series, start, end, context, horizon, stride,
                                          test_start_window, test_windows, batch):
            if warmed >= 1:
                break
            _ = forecast_timesfm_point(tfm, Xb[:1], horizon)
            warmed += 1

        for Xb, Yb in _iter_window_batches(series, start, end, context, horizon, stride,
                                          test_start_window, test_windows, batch):
            t0 = time.perf_counter()
            Pb = forecast_timesfm_point(tfm, Xb, horizon)
            t1 = time.perf_counter()
            if measure_time:
                times.append(t1 - t0)

            d = (Pb - Yb).astype(np.float64)
            agg.sse += float(np.sum(d * d))
            agg.sae += float(np.sum(np.abs(d)))
            agg.n += int(d.size)

        mse_s, mae_s = mse_mae_from_agg(agg)
        per_series_mse.append(mse_s)
        per_series_mae.append(mae_s)

        micro.sse += agg.sse
        micro.sae += agg.sae
        micro.n += agg.n

    macro_mse = float(np.mean(per_series_mse))
    macro_mae = float(np.mean(per_series_mae))
    micro_mse, micro_mae = mse_mae_from_agg(micro)
    avg_batch_sec = float(np.mean(times)) if (measure_time and times) else 0.0
    return (macro_mse, macro_mae, micro_mse, micro_mae, avg_batch_sec)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)

    # Multi-col selection
    ap.add_argument("--cols_regex", type=str, default=None,
                    help="Regex to select multiple target columns, e.g. '^MT_\\d+$'")
    ap.add_argument("--cols", type=str, default=None,
                    help="Comma-separated list of columns (overrides cols_regex).")
    ap.add_argument("--drop_cols_regex", type=str, default=r"^(date|timestamp)$",
                    help="Regex to drop cols even if matched (default drops 'date').")

    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--train_end", type=int, default=49152)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_test", type=int, default=96)

    ap.add_argument("--calib_windows", type=int, default=1091,
                    help="Total calibration pool windows across ALL series (not per-series).")
    ap.add_argument("--calib_pool_mode", type=str, default="first", choices=["first", "random"],
                    help="How to build the multi-series calibration pool.")
    ap.add_argument("--calib_select", type=str, default="first", choices=["first", "random", "topk"],
                    help="Select K from pool for gram collection. K is compute budget.")
    ap.add_argument("--weight_norm", type=str, default="per_series", choices=["global", "per_series"],
                    help="How to normalize error weights across many series.")

    ap.add_argument("--test_windows", type=int, default=-1, help="-1 uses all test windows per series.")
    ap.add_argument("--test_start_window", type=int, default=0)

    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--calib_batch", type=int, default=4)

    ap.add_argument("--include_quantile_head", action="store_true")
    ap.add_argument("--include_regex", type=str, default=".*")
    ap.add_argument("--exclude_regex", type=str, default="")
    ap.add_argument("--sample_rows_per_call", type=int, default=2048)
    ap.add_argument("--max_calls_per_layer", type=int, default=32)

    ap.add_argument("--score_mode", type=str, default="ratio", choices=["ratio", "keep"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-5)

    ap.add_argument("--model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    ap.add_argument("--error_power", type=float, default=1.0,
                    help="Exponent for error weighting. 0 => uniform (label-free if calib_select != topk).")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--measure_time", action="store_true")
    args = ap.parse_args()

    # Load df + columns
    df, cols = load_multiseries(args.csv, args.cols_regex, args.cols, args.drop_cols_regex)
    n_total = len(df)
    S = len(cols)

    # basic feasibility
    min_req = args.context + args.horizon + 1
    if args.train_end < min_req:
        raise ValueError(f"train_end too small: need >= {min_req}")
    if (n_total - args.train_end) < min_req:
        raise ValueError(f"test segment too small: need n_total-train_end >= {min_req}")

    # Counts (per series) for logging
    n_train_w = _count_windows(0, args.train_end, args.context, args.horizon, args.stride_train)
    n_test_w  = _count_windows(args.train_end, n_total, args.context, args.horizon, args.stride_test)

    print(f"[data] total_rows={n_total} n_series={S}")
    print(f"[split] train_end={args.train_end} context={args.context} horizon={args.horizon}")
    print(f"[train] windows_per_series={n_train_w} stride_train={args.stride_train}")
    print(f"[test]  windows_per_series={n_test_w} stride_test={args.stride_test} start_window={args.test_start_window} req_windows={args.test_windows}")
    gram_budget = args.max_calls_per_layer * args.calib_batch
    eff_K = min(args.calib_windows, gram_budget)
    print(f"[calib] pool_total={args.calib_windows} pool_mode={args.calib_pool_mode} select={args.calib_select}")
    print(f"[calib] gram_budget=max_calls_per_layer*calib_batch={args.max_calls_per_layer}*{args.calib_batch}={gram_budget} => effective_K={eff_K}")
    print(f"[calib] weight_norm={args.weight_norm} error_power={args.error_power}")

    # Model
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    torch_mod = find_torch_module(tfm)
    targets = select_linears(torch_mod, args.include_quantile_head, args.include_regex, args.exclude_regex)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=max(args.horizon, 256)))
    print(f"[model] targets={len(targets)}")

    # Baseline eval (macro + micro)
    macro_mse_b, macro_mae_b, micro_mse_b, micro_mae_b, tsec = eval_all_series(
        tfm, df, cols,
        start=args.train_end, end=n_total,
        context=args.context, horizon=args.horizon, stride=args.stride_test,
        test_start_window=args.test_start_window, test_windows=args.test_windows,
        batch=args.batch, measure_time=args.measure_time
    )
    if args.measure_time:
        print(f"[baseline] MACRO MSE={macro_mse_b:.6f} MAE={macro_mae_b:.6f} | MICRO MSE={micro_mse_b:.6f} MAE={micro_mae_b:.6f} | avg_batch_sec={tsec:.4f}")
    else:
        print(f"[baseline] MACRO MSE={macro_mse_b:.6f} MAE={macro_mae_b:.6f} | MICRO MSE={micro_mse_b:.6f} MAE={micro_mae_b:.6f}")

    # Build calibration pool (total windows across ALL series)
    X_pool, Y_pool, sid_pool = build_calib_pool(
        df=df, cols=cols, pool_size=args.calib_windows,
        train_end=args.train_end, context=args.context, horizon=args.horizon,
        stride_train=args.stride_train, seed=args.seed, mode=args.calib_pool_mode
    )
    print(f"[calib] built pool: X_pool={X_pool.shape} (total windows), unique_series_in_pool={len(np.unique(sid_pool))}/{S}")

    # Need errors?
    need_errors = (args.calib_select == "topk") or (args.error_power != 0.0)
    if need_errors:
        print(f"[calib] computing errors/weights on pool (batched={args.calib_batch}) ...")
        errors, weights = compute_errors_and_weights(
            tfm_model=tfm,
            X_pool=X_pool,
            Y_pool=Y_pool,
            sid_pool=sid_pool,
            horizon=args.horizon,
            calib_batch=args.calib_batch,
            error_power=args.error_power,
            weight_norm=args.weight_norm,
        )
        print(f"[calib] weight_stats: min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f}")
    else:
        errors = None
        weights = np.ones((X_pool.shape[0],), dtype=np.float32)
        print(f"[calib] uniform weights (no labels used): power=0, select={args.calib_select}")

    # Select K from pool
    rng = np.random.default_rng(args.seed)
    K = min(eff_K, X_pool.shape[0])

    if args.calib_select == "first":
        sel_idx = np.arange(K, dtype=np.int64)
    elif args.calib_select == "random":
        sel_idx = rng.choice(X_pool.shape[0], size=K, replace=False).astype(np.int64)
    elif args.calib_select == "topk":
        if errors is None:
            raise RuntimeError("topk selection requires errors.")
        sel_idx = np.argsort(errors)[-K:].astype(np.int64)
    else:
        raise ValueError(f"Unknown calib_select: {args.calib_select}")

    X_sel = X_pool[sel_idx]
    w_sel = weights[sel_idx].astype(np.float32)

    print(f"[calib] collecting grams: K={K} (from pool={X_pool.shape[0]})")

    gram_stats = collect_group_grams_weighted(
        tfm_model=tfm,
        targets=targets,
        X_sel=X_sel,
        weights_sel=w_sel,
        horizon=args.horizon,
        calib_batch=args.calib_batch,
        sample_rows_per_call=args.sample_rows_per_call,
        max_calls_per_layer=args.max_calls_per_layer,
    )
    print(f"[calib] collected grams for {len(gram_stats)}/{len(targets)} layers")

    # Prune
    print(f"[prune] SNR 2:4: score_mode={args.score_mode}, refit={bool(args.refit)} ridge={args.ridge:g}")
    for name, layer in targets:
        st = gram_stats.get(name, None)
        if st is None or st.Csig <= 0.0:
            continue
        prune_linear_snr_2of4(layer, st.Gsum, st.count, args.score_mode, args.eps, bool(args.refit), args.ridge)

    # Eval pruned
    macro_mse_p, macro_mae_p, micro_mse_p, micro_mae_p, tsec2 = eval_all_series(
        tfm, df, cols,
        start=args.train_end, end=n_total,
        context=args.context, horizon=args.horizon, stride=args.stride_test,
        test_start_window=args.test_start_window, test_windows=args.test_windows,
        batch=args.batch, measure_time=args.measure_time
    )
    if args.measure_time:
        print(f"[snr-2of4] MACRO MSE={macro_mse_p:.6f} MAE={macro_mae_p:.6f} | MICRO MSE={micro_mse_p:.6f} MAE={micro_mae_p:.6f} | avg_batch_sec={tsec2:.4f}")
    else:
        print(f"[snr-2of4] MACRO MSE={macro_mse_p:.6f} MAE={macro_mae_p:.6f} | MICRO MSE={micro_mse_p:.6f} MAE={micro_mae_p:.6f}")

    print(f"[delta] ΔMACRO MSE={(macro_mse_p - macro_mse_b):+.6f}  ΔMACRO MAE={(macro_mae_p - macro_mae_b):+.6f}")
    print(f"[delta] ΔMICRO MSE={(micro_mse_p - micro_mse_b):+.6f}  ΔMICRO MAE={(micro_mae_p - micro_mae_b):+.6f}")

if __name__ == "__main__":
    main()
