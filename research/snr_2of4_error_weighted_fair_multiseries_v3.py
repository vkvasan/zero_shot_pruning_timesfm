#!/usr/bin/env python3
"""
snr_2of4_error_weighted_fair_multiseries_v3.py

TimesFM strict 2:4 error-weighted SNR pruning for multi-series CSVs (e.g., Electricity).
Adds Chronos-style fast evaluation via --eval_mode last (one forecast per series).

Key ideas:
- Calibration uses a subset of columns (--calib_cols) and a pool of training windows (rolling).
- Gram collection uses budget K = max_calls_per_layer * calib_batch (per target layer).
- Evaluation:
    * rolling: macro-average over columns and rolling test windows (bounded by --test_windows)
    * last:    macro-average over columns using ONLY the last available test window (Chronos-style fast)
- Prints effective sparsity across targeted Linear weights.

This is zero-shot in the sense that no backprop is used; calibration uses train labels only to form error weights.
"""
import argparse, re, time
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
# Data helpers
# -------------------------
def load_multiseries(csv_path: str, cols_regex: str, drop_date: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if drop_date and 'date' in df.columns:
        df = df.drop(columns=['date'])
    df = df.select_dtypes(include=[np.number])
    if cols_regex:
        df = df.filter(regex=cols_regex)
    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns matched cols_regex={cols_regex!r} in {csv_path}")
    return df

def make_windows_1d(series: np.ndarray, start: int, end: int, context: int, horizon: int, stride: int):
    xs, ys = [], []
    last = end - (context + horizon)
    for i in range(start, last + 1, stride):
        xs.append(series[i:i+context])
        ys.append(series[i+context:i+context+horizon])
    if not xs:
        raise ValueError("No windows produced (check split/context/horizon/stride).")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)

def mse_mae(pred: np.ndarray, tgt: np.ndarray):
    d = pred - tgt
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))

# -------------------------
# Target selection
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
        if m.in_features < 4:
            continue
        out.append((name, m))
    return out

# -------------------------
# Gram collection (grouped 4)
# -------------------------
@dataclass
class GramStat:
    Gsum: torch.Tensor   # [G,4,4]
    count: float         # effective weight sum

@torch.no_grad()
def compute_errors_and_weights(
    tfm_model,
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    horizon: int,
    calib_batch: int,
    error_power: float,
):
    preds = []
    for i in range(0, len(X_pool), calib_batch):
        preds.append(forecast_timesfm_point(tfm_model, X_pool[i:i+calib_batch], horizon))
    preds = np.concatenate(preds, axis=0)
    diff = preds - Y_pool
    errors = np.mean(diff**2, axis=1).astype(np.float32)
    mean_err = float(np.mean(errors))
    err_ratio = (errors / (mean_err + 1e-6)).astype(np.float32)
    weights = (err_ratio ** float(error_power)).astype(np.float32)
    return errors, err_ratio, weights

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
                xf = x.reshape(-1, x.shape[-1])  # [B*T,C]
                w_exp = w_batch.unsqueeze(1).expand(B, T).reshape(-1)  # [B*T]
            else:
                xf = x
                w_exp = w_batch

            G = xf.shape[-1] // 4
            Cg = G * 4
            if Cg == 0:
                return
            xf = xf[:, :Cg]

            if xf.shape[0] > sample_rows_per_call:
                idx = torch.randint(0, xf.shape[0], (sample_rows_per_call,), device=xf.device)
                xf = xf.index_select(0, idx)
                w_exp = w_exp.index_select(0, idx)

            xg = xf.reshape(xf.shape[0], G, 4)  # [N,G,4]
            G_batch = torch.einsum("n,ngc,ngd->gcd", w_exp, xg, xg).cpu()
            c_batch = float(w_exp.sum().item())

            if layer_name not in stats:
                stats[layer_name] = GramStat(Gsum=G_batch, count=c_batch)
            else:
                st = stats[layer_name]
                st.Gsum += G_batch
                st.count += c_batch

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
# SNR pruning (2:4)
# -------------------------
PAIRS = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.long)
PAIR_MASKS = torch.zeros((6,4), dtype=torch.float32)
for k in range(6):
    i,j = PAIRS[k].tolist()
    PAIR_MASKS[k,i] = PAIR_MASKS[k,j] = 1.0

@torch.no_grad()
def prune_linear_snr_2of4(layer, st: GramStat, score_mode: str, eps: float, refit: bool, ridge: float):
    W = layer.weight.data
    O, C = W.shape
    Ggroups = C // 4
    Cg = Ggroups * 4
    if Cg == 0:
        return
    device = W.device
    dtype = W.dtype

    Wg = W[:, :Cg].view(O, Ggroups, 4)
    G = (st.Gsum / max(st.count, 1e-6)).to(device=device, dtype=dtype)  # [G,4,4]

    masks = PAIR_MASKS.to(device=device, dtype=dtype)
    scores = torch.empty((O, Ggroups, 6), device=device, dtype=dtype)

    for k in range(6):
        mk = masks[k].view(1, 1, 4)
        md = (1.0 - mk)
        Wk = Wg * mk
        Wd = Wg * md

        Tk = torch.einsum("ogc,gcd->ogd", Wk, G)
        Ek = (Tk * Wk).sum(dim=2)
        if score_mode == "keep":
            scores[:, :, k] = Ek
        else:
            Td = torch.einsum("ogc,gcd->ogd", Wd, G)
            Ed = (Td * Wd).sum(dim=2)
            scores[:, :, k] = Ek / (Ed + eps)

    bestk = torch.argmax(scores, dim=2)
    pair_idx = PAIRS.to(device=device)[bestk]
    keep_mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool)
    keep_mask.scatter_(2, pair_idx, torch.ones_like(pair_idx, dtype=torch.bool))

    if not refit:
        Wnew = torch.where(keep_mask, Wg, torch.zeros_like(Wg))
    else:
        B = torch.einsum("ogc,gcd->ogd", Wg, G)  # [O,G,4]
        invs = []
        eye2 = torch.eye(2, device=device, dtype=dtype).view(1, 2, 2)
        for k in range(6):
            i, j = PAIRS[k].tolist()
            Gss = G[:, [i, j]][:, :, [i, j]] + ridge * eye2
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

def compute_effective_sparsity(targets: List[Tuple[str, nn.Linear]]) -> Tuple[float, int, int]:
    zero = 0
    total = 0
    for _name, layer in targets:
        W = layer.weight.data
        C = W.shape[1]
        Cg = (C // 4) * 4
        if Cg == 0:
            continue
        block = W[:, :Cg]
        zero += int((block == 0).sum().item())
        total += int(block.numel())
    return (zero / total) if total else 0.0, zero, total

# -------------------------
# Column selection
# -------------------------
def pick_columns(cols: List[str], k: int, mode: str, seed: int) -> List[str]:
    if k < 0 or k >= len(cols):
        return list(cols)
    rng = np.random.default_rng(seed)
    if mode == "first":
        return list(cols[:k])
    if mode == "random":
        idx = rng.choice(len(cols), size=k, replace=False)
        return [cols[i] for i in idx]
    raise ValueError(f"Unknown col select mode: {mode}")

# -------------------------
# Evaluation modes
# -------------------------
def eval_last_window(
    tfm_model,
    df: pd.DataFrame,
    cols: List[str],
    train_end: int,
    context: int,
    horizon: int,
    batch: int,
    measure_time: bool,
):
    Xs, Ys = [], []
    for c in cols:
        s = df[c].to_numpy(dtype=np.float32)
        if len(s) < train_end + context + horizon:
            continue
        st = len(s) - (context + horizon)
        if st < train_end:
            st = train_end
        x = s[st:st+context]
        y = s[st+context:st+context+horizon]
        if x.shape[0] == context and y.shape[0] == horizon:
            Xs.append(x); Ys.append(y)
    if not Xs:
        raise RuntimeError("No series produced a valid last-window sample (check split/context/horizon).")
    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)
    if measure_time:
        pred, tsec = timed_forecast(tfm_model, X, horizon, batch)
        mse, mae = mse_mae(pred, Y)
        return mse, mae, tsec
    preds = []
    for i in range(0, len(X), batch):
        preds.append(forecast_timesfm_point(tfm_model, X[i:i+batch], horizon))
    pred = np.concatenate(preds, axis=0)
    mse, mae = mse_mae(pred, Y)
    return mse, mae, 0.0

def eval_rolling(
    tfm_model,
    df: pd.DataFrame,
    cols: List[str],
    train_end: int,
    context: int,
    horizon: int,
    stride_test: int,
    test_windows: int,
    test_start_window: int,
    batch: int,
    measure_time: bool,
):
    mses, maes, times = [], [], []
    for c in cols:
        s = df[c].to_numpy(dtype=np.float32)
        if len(s) < train_end + context + horizon:
            continue
        X_all, Y_all = make_windows_1d(s, train_end, len(s), context, horizon, stride_test)
        ts = max(0, test_start_window)
        if test_windows < 0:
            te = X_all.shape[0]
        else:
            te = min(ts + test_windows, X_all.shape[0])
        X = X_all[ts:te]
        Y = Y_all[ts:te]
        if X.shape[0] == 0:
            continue
        if measure_time:
            pred, tsec = timed_forecast(tfm_model, X, horizon, batch)
            mse, mae = mse_mae(pred, Y)
            times.append(tsec)
        else:
            preds = []
            for i in range(0, len(X), batch):
                preds.append(forecast_timesfm_point(tfm_model, X[i:i+batch], horizon))
            pred = np.concatenate(preds, axis=0)
            mse, mae = mse_mae(pred, Y)
        mses.append(mse)
        maes.append(mae)
    if not mses:
        raise RuntimeError("No test windows evaluated (check split/filters).")
    mse = float(np.mean(mses))
    mae = float(np.mean(maes))
    tsec = float(np.mean(times)) if times else 0.0
    return mse, mae, tsec

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--cols_regex", type=str, default=r"^MT_\d+$")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_test", type=int, default=96)

    ap.add_argument("--calib_cols", type=int, default=64)
    ap.add_argument("--calib_cols_select", type=str, default="random", choices=["first","random"])
    ap.add_argument("--eval_cols", type=int, default=-1, help="-1 means all matched columns")
    ap.add_argument("--eval_cols_select", type=str, default="random", choices=["first","random"])

    ap.add_argument("--calib_windows", type=int, default=512)
    ap.add_argument("--calib_select", type=str, default="random", choices=["first","random","topk"])

    ap.add_argument("--test_windows", type=int, default=64, help="-1 uses all available (rolling mode only)")
    ap.add_argument("--test_start_window", type=int, default=0)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--calib_batch", type=int, default=4)

    ap.add_argument("--include_quantile_head", action="store_true")
    ap.add_argument("--include_regex", type=str, default=r"stacked_xf\..*\.(ff0|ff1)$")
    ap.add_argument("--exclude_regex", type=str, default="")
    ap.add_argument("--sample_rows_per_call", type=int, default=1024)
    ap.add_argument("--max_calls_per_layer", type=int, default=32)

    ap.add_argument("--score_mode", type=str, default="ratio", choices=["ratio","keep"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-5)

    ap.add_argument("--model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    ap.add_argument("--error_power", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--measure_time", action="store_true")

    ap.add_argument("--eval_mode", type=str, default="rolling", choices=["rolling","last"])

    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"[device] torch.cuda.is_available()={use_cuda}")
    if use_cuda:
        print(f"[device] cuda_device={torch.cuda.get_device_name(0)}")

    df = load_multiseries(args.csv, args.cols_regex)
    cols_all = list(df.columns)
    n_total = len(df)
    train_end = int(n_total * float(args.train_frac))
    print(f"[data] rows={n_total} cols={len(cols_all)} train_end={train_end} train_frac={args.train_frac:g}")
    print(f"[cols] regex={args.cols_regex!r}")

    calib_cols = pick_columns(cols_all, args.calib_cols, args.calib_cols_select, args.seed)
    eval_cols = pick_columns(cols_all, args.eval_cols, args.eval_cols_select, args.seed + 1)
    print(f"[cols] calib_cols={len(calib_cols)} eval_cols={len(eval_cols)} eval_mode={args.eval_mode}")

    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    torch_mod = find_torch_module(tfm)
    targets = select_linears(torch_mod, args.include_quantile_head, args.include_regex, args.exclude_regex)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=max(args.horizon, 256)))
    print(f"[targets] linears_selected={len(targets)} include_regex={args.include_regex!r}")

    # Baseline eval
    if args.eval_mode == "last":
        mse_b, mae_b, tsec = eval_last_window(tfm, df, eval_cols, train_end, args.context, args.horizon, args.batch, args.measure_time)
    else:
        mse_b, mae_b, tsec = eval_rolling(tfm, df, eval_cols, train_end, args.context, args.horizon, args.stride_test,
                                         args.test_windows, args.test_start_window, args.batch, args.measure_time)
    if args.measure_time:
        print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f} | avg_batch_sec={tsec:.4f}")
    else:
        print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f}")

    # Calibration pool stacked over columns
    X_pool_list, Y_pool_list = [], []
    for c in calib_cols:
        s = df[c].to_numpy(dtype=np.float32)
        if len(s) < train_end + args.context + args.horizon:
            continue
        X_train, Y_train = make_windows_1d(s, 0, train_end, args.context, args.horizon, args.stride_train)
        if X_train.shape[0] == 0:
            continue
        pool = min(args.calib_windows, X_train.shape[0])
        X_pool_list.append(X_train[:pool])
        Y_pool_list.append(Y_train[:pool])
    if not X_pool_list:
        raise RuntimeError("Calibration pool is empty (check calib_cols/train_frac/context/horizon).")
    X_pool = np.concatenate(X_pool_list, axis=0)
    Y_pool = np.concatenate(Y_pool_list, axis=0)
    print(f"[calib] pool_windows_total={X_pool.shape[0]} (per-col capped at {args.calib_windows})")

    gram_budget = args.max_calls_per_layer * args.calib_batch
    K = min(X_pool.shape[0], gram_budget)
    print(f"[calib] calib_batch={args.calib_batch} gram_budget={args.max_calls_per_layer}*{args.calib_batch}={gram_budget} => effective_K={K}")

    need_errors = (args.calib_select == "topk") or (args.error_power != 0.0)
    if need_errors:
        print(f"[calib] computing errors/weights on pool: power={args.error_power} (batched={args.calib_batch}) ...")
        errors, err_ratio, weights = compute_errors_and_weights(tfm, X_pool, Y_pool, args.horizon, args.calib_batch, args.error_power)
        print(f"[calib] weight_stats: min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f} (power={args.error_power})")
    else:
        errors = None
        weights = np.ones((X_pool.shape[0],), dtype=np.float32)
        print(f"[calib] uniform weights (no labels used): power=0, select={args.calib_select}")

    rng = np.random.default_rng(args.seed)
    if args.calib_select == "first":
        sel_idx = np.arange(K, dtype=np.int64)
    elif args.calib_select == "random":
        sel_idx = rng.choice(X_pool.shape[0], size=K, replace=False).astype(np.int64)
    else:
        if errors is None:
            raise RuntimeError("topk selection requires errors.")
        sel_idx = np.argsort(errors)[-K:].astype(np.int64)

    X_sel = X_pool[sel_idx]
    w_sel = weights[sel_idx].astype(np.float32)

    print(f"[calib] collecting grams: select={args.calib_select} pool={X_pool.shape[0]} K={K} targets={len(targets)}")
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

    print(f"[prune] SNR 2:4: score_mode={args.score_mode}, refit={bool(args.refit)} ridge={args.ridge:g}")
    for name, layer in targets:
        st = gram_stats.get(name, None)
        if st is None or st.count <= 0:
            continue
        prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), args.ridge)

    s_eff, zc, tc = compute_effective_sparsity(targets)
    print(f"[sparsity] targeted_linear_region_zeros={zc} / {tc} => effective_sparsity={100.0*s_eff:.2f}% (2:4 ideal is 50.00%)")

    if args.eval_mode == "last":
        mse_p, mae_p, tsec2 = eval_last_window(tfm, df, eval_cols, train_end, args.context, args.horizon, args.batch, args.measure_time)
    else:
        mse_p, mae_p, tsec2 = eval_rolling(tfm, df, eval_cols, train_end, args.context, args.horizon, args.stride_test,
                                           args.test_windows, args.test_start_window, args.batch, args.measure_time)
    if args.measure_time:
        print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f} | avg_batch_sec={tsec2:.4f}")
    else:
        print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f}")
    print(f"[delta] ΔMSE={(mse_p - mse_b):+.6f}  ΔMAE={(mae_p - mae_b):+.6f}")

if __name__ == "__main__":
    main()
