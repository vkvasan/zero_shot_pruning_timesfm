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
    # For each target Linear layer, we optionally keep two sets of grouped grams:
    #   - Gsig: emphasizes "predictable / low-error" behavior
    #   - Gnoi: emphasizes "hard / high-error" behavior
    # If you use score_mode in {"keep","ratio"}, only Gsig is used (and equals the legacy weighted gram).
    Gsig: torch.Tensor
    Csig: float
    Gnoi: Optional[torch.Tensor] = None
    Cnoi: float = 0.0


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
      errors:  [N] MSE per window
      err_ratio: [N] errors / mean(errors)
      weights: [N] normalized weights = (err_ratio)^error_power
    """
    preds = []
    for i in range(0, len(X_pool), calib_batch):
        preds.append(forecast_timesfm_point(tfm_model, X_pool[i:i+calib_batch], horizon))
    preds = np.concatenate(preds, axis=0)  # [N, H]
    diff = preds - Y_pool
    errors = np.mean(diff**2, axis=1).astype(np.float32)  # [N]
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
    w_sig_sel: np.ndarray,            # [K] per-window weights for "signal" gram
    w_noi_sel: Optional[np.ndarray],  # [K] per-window weights for "noise" gram (or None to disable)
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
    - Keep calib_batch small enough that TimesFM doesn't internally split it, otherwise weight alignment becomes best-effort.
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

            ws_batch = current_batch_wsig.to(x.device)  # expected [B]
            wn_batch = None if current_batch_wnoi is None else current_batch_wnoi.to(x.device)

            # If TimesFM internally micro-batches, B may be smaller than requested.
            if ws_batch.numel() != B:
                if not warned_split:
                    print(f"[warn] calib batch mismatch inside model: weights={ws_batch.numel()} but hook sees B={B}. "
                          f"TimesFM likely micro-batched. Set --calib_batch smaller (e.g., 4 or 1).")
                    warned_split = True
                if ws_batch.numel() >= B:
                    ws_batch = ws_batch[:B]
                    if wn_batch is not None:
                        wn_batch = wn_batch[:B] if wn_batch.numel() >= B else torch.cat(
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
                ws_exp = ws_batch.unsqueeze(1).expand(B, T).reshape(-1)  # [B*T]
                wn_exp = None if wn_batch is None else wn_batch.unsqueeze(1).expand(B, T).reshape(-1)
            else:
                xf = x
                ws_exp = ws_batch
                wn_exp = wn_batch

            G = xf.shape[-1] // 4
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

            xg = xf.reshape(xf.shape[0], G, 4)  # [N,G,4]

            # Weighted gram per group: sum_n w_n * x_n x_n^T
            Gs_batch = torch.einsum("n,ngc,ngd->gcd", ws_exp, xg, xg).cpu()
            Cs = float(ws_exp.sum().item())

            Gn_batch = None
            Cn = 0.0
            if wn_exp is not None:
                Gn_batch = torch.einsum("n,ngc,ngd->gcd", wn_exp, xg, xg).cpu()
                Cn = float(wn_exp.sum().item())

            if layer_name not in stats:
                stats[layer_name] = GramStat(Gsig=Gs_batch, Csig=Cs, Gnoi=Gn_batch, Cnoi=Cn)
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

            calls[layer_name] += 1
        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    # drive forward passes (for hooks)
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
    i,j = PAIRS[k].tolist()
    PAIR_MASKS[k,i] = PAIR_MASKS[k,j] = 1.0

@torch.no_grad()
def prune_linear_snr_2of4(layer, st: GramStat, score_mode: str, eps: float, refit: bool, ridge: float):
    """Strict 2:4 pruning for a Linear layer.

    score_mode:
      - "keep": keep-energy only (legacy)
      - "ratio": keep/drop ratio on a single gram (legacy)
      - "sn_ratio2": ratio-of-ratios using signal/noise grams (new, forward-only)
          Rs = E_sig_keep / (E_sig_drop + eps)
          Rn = E_noi_keep / (E_noi_drop + eps)
          score = Rs / (Rn + eps)
        If noise grams are not available, falls back to "ratio" on signal gram.
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
    # Normalize grams by their effective weighted counts (so the absolute scale doesn't dominate).
    Gs = (st.Gsig / max(st.Csig, 1e-6)).to(device=device, dtype=dtype)

    use_noise = (score_mode == "sn_ratio2") and (st.Gnoi is not None) and (st.Cnoi > 0.0)
    if use_noise:
        Gn = (st.Gnoi / max(st.Cnoi, 1e-6)).to(device=device, dtype=dtype)
    else:
        Gn = None
        if score_mode == "sn_ratio2":
            # graceful fallback
            score_mode = "ratio"

    masks = PAIR_MASKS.to(device=device, dtype=dtype)
    scores = torch.empty((O, Ggroups, 6), device=device, dtype=dtype)

    for k in range(6):
        mk = masks[k].view(1, 1, 4)
        md = (1.0 - mk)
        Wk = Wg * mk
        Wd = Wg * md

        # --- signal energies ---
        Tk_s = torch.einsum("ogc,gcd->ogd", Wk, Gs)
        Ek_s = (Tk_s * Wk).sum(dim=2)  # [O,G]
        Td_s = torch.einsum("ogc,gcd->ogd", Wd, Gs)
        Ed_s = (Td_s * Wd).sum(dim=2)

        if score_mode == "keep":
            scores[:, :, k] = Ek_s
        elif score_mode == "ratio":
            scores[:, :, k] = Ek_s / (Ed_s + eps)
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
        # Refit with the *signal* gram (we want to match predictable structure)
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
            gsel = g_idx[sel]                    # [N]
            invsel = invs[k, gsel, :, :]         # [N,2,2]
            u = torch.bmm(invsel, bsel.unsqueeze(2)).squeeze(2)
            Wnew[:, :, i][sel] = u[:, 0]
            Wnew[:, :, j][sel] = u[:, 1]

    W[:, :Cg] = Wnew.view(O, Cg)
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

    ap.add_argument("--score_mode", type=str, default="sn_ratio2", choices=["sn_ratio2", "ratio", "keep"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-5)

    ap.add_argument("--model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    ap.add_argument("--error_power", type=float, default=1.0,
                    help="Exponent for error weighting. Set 0 for uniform weights (label-free if calib_select != topk).")
    ap.add_argument("--sn_gamma", type=float, default=1.0,
                    help="For score_mode=sn_ratio2: build two grams using err_ratio^(-sn_gamma) as 'signal' weights and err_ratio^(sn_gamma) as 'noise' weights (forward-only).")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--measure_time", action="store_true")
    args = ap.parse_args()

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

    # Logs for fairness/transparency
    print(f"[split] total_rows={n_total} train_end={args.train_end} context={args.context} horizon={args.horizon}")
    print(f"[train] windows={X_train.shape[0]} calib_pool={X_pool.shape[0]}")
    req_tw = args.test_windows
    print(f"[test]  available={X_test_all.shape[0]} requested={req_tw} start={ts} using={X_test.shape[0]} stride_test={args.stride_test}")
    gram_budget = args.max_calls_per_layer * args.calib_batch
    eff_K = min(X_pool.shape[0], gram_budget)
    print(f"[calib] eval_batch={args.batch} calib_batch={args.calib_batch} gram_budget=max_calls_per_layer*calib_batch={args.max_calls_per_layer}*{args.calib_batch}={gram_budget} => effective_K={eff_K}")

    # Model
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    torch_mod = find_torch_module(tfm)
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
            tfm_model=tfm, X_pool=X_pool, Y_pool=Y_pool,
            horizon=args.horizon, calib_batch=args.calib_batch, error_power=args.error_power
        )
        print(f"[calib] weight_stats: min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f} (power={args.error_power})")
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
        # top-K by error (equivalent to top-K by weight when power>0)
        if errors is None:
            raise RuntimeError("topk selection requires errors, but errors were not computed.")
        sel_idx = np.argsort(errors)[-K:].astype(np.int64)
    else:
        raise ValueError(f"Unknown calib_select: {args.calib_select}")

    X_sel = X_pool[sel_idx]

    # --- weights used for gram collection ---
    # Legacy behavior (single gram): w_sel = (err_ratio^error_power) over selected windows.
    # New behavior for score_mode=sn_ratio2: build *two* grams:
    #   signal weights  ~ err_ratio^(-sn_gamma)  (emphasize easy/predictable windows)
    #   noise weights   ~ err_ratio^(+sn_gamma)  (emphasize hard/high-error windows)
    w_sel = weights[sel_idx].astype(np.float32)
    er_sel = err_ratio[sel_idx].astype(np.float32)

    if args.score_mode == "sn_ratio2":
        ws = (er_sel ** (-float(args.sn_gamma))).astype(np.float32)
        wn = (er_sel ** ( float(args.sn_gamma))).astype(np.float32)
        # normalize to mean 1 for numerical stability / comparable counts
        ws = (ws / (ws.mean() + 1e-8)).astype(np.float32)
        wn = (wn / (wn.mean() + 1e-8)).astype(np.float32)
        w_sig_sel, w_noi_sel = ws, wn
        print(f"[calib] sn_ratio2 grams: sn_gamma={args.sn_gamma:g} | "
              f"sig_w(min/mean/max)=({ws.min():.3g}/{ws.mean():.3g}/{ws.max():.3g}) "
              f"noi_w(min/mean/max)=({wn.min():.3g}/{wn.mean():.3g}/{wn.max():.3g})")
    else:
        w_sig_sel, w_noi_sel = w_sel, None

    print(f"[calib] collecting grams: select={args.calib_select} pool={X_pool.shape[0]} K={K} (max_calls_per_layer={args.max_calls_per_layer})")

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

    # Prune
    print(f"[prune] SNR 2:4: score_mode={args.score_mode}, refit={bool(args.refit)} ridge={args.ridge:g}")
    for name, layer in targets:
        st = gram_stats.get(name, None)
        if st is None or st.Csig <= 0.0:
            continue
        prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), args.ridge)

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
