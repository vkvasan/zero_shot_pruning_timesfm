#!/usr/bin/env python3
"""
snr_2of4_error_weighted_fair_multiseries_v2.py

Adds FAST proxy evaluation for multi-series datasets (e.g., Electricity MT_1..MT_320):

- Calibration/pruning happens ONCE (same as v1):
    * In multi-series mode, build calibration pool by concatenating windows from a subset of columns
      (--calib_cols, --calib_cols_select).

- Evaluation (baseline + pruned) can be:
    * FULL:  macro-average over ALL matched columns (eval_cols=-1)
    * PROXY: macro-average over a random/first subset of columns (eval_cols=N)

Example FAST Electricity run:
python snr_2of4_error_weighted_fair_multiseries_v2.py \
  --csv electricity/electricity.csv --cols_regex '^MT_\\d+$' \
  --train_frac 0.7 --context 1024 --horizon 96 \
  --stride_train 1 --stride_test 96 \
  --calib_cols 32 --calib_cols_select random \
  --calib_windows 256 --calib_select random \
  --eval_cols 32 --eval_cols_select random \
  --test_windows 16 \
  --batch 16 --calib_batch 4 \
  --include_regex "stacked_xf\\..*\\.(ff0|ff1)$" \
  --sample_rows_per_call 1024 --max_calls_per_layer 32 \
  --score_mode ratio --refit 1 --ridge 1e-5 \
  --error_power 1.0 --seed 2026 --measure_time

For FINAL Electricity numbers (slow):
  --eval_cols -1 --test_windows -1
"""

import argparse
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def load_series(csv_path: str, col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in {csv_path}.")
    return df[col].to_numpy(dtype=np.float32)


def load_series_matrix(csv_path: str, cols_regex: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    pat = re.compile(cols_regex)
    cols = [c for c in df.columns if pat.match(c)]
    if not cols:
        raise ValueError(f"No columns matched regex {cols_regex} in {csv_path}.")
    mat = df[cols].to_numpy(dtype=np.float32)
    return mat, cols


def make_windows_1d(series: np.ndarray, start: int, end: int, context: int, horizon: int, stride: int):
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


@dataclass
class GramStat:
    Gsig: torch.Tensor
    Csig: float


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
    weights = (err_ratio) ** float(error_power)
    return errors, err_ratio, weights.astype(np.float32)


@torch.no_grad()
def collect_group_grams(
    tfm_model,
    targets: List[Tuple[str, nn.Linear]],
    X_sel: np.ndarray,
    w_sel: np.ndarray,
    horizon: int,
    calib_batch: int,
    sample_rows_per_call: int,
    max_calls_per_layer: int,
):
    stats: Dict[str, GramStat] = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []

    global current_batch_w
    current_batch_w = None
    warned_split = False

    def make_hook(layer_name: str):
        def pre_hook(_mod, inputs):
            nonlocal warned_split
            if calls[layer_name] >= max_calls_per_layer:
                return
            (x,) = inputs
            B = x.shape[0]
            if current_batch_w is None:
                raise RuntimeError("Internal error: current_batch_w not set.")
            w_batch = current_batch_w.to(x.device)

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
                w_exp = w_batch.unsqueeze(1).expand(B, T).reshape(-1)
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

            xg = xf.reshape(xf.shape[0], G, 4)
            G_batch = torch.einsum("n,ngc,ngd->gcd", w_exp, xg, xg).cpu()
            C_batch = float(w_exp.sum().item())

            if layer_name not in stats:
                stats[layer_name] = GramStat(Gsig=G_batch, Csig=C_batch)
            else:
                st = stats[layer_name]
                st.Gsig += G_batch
                st.Csig += C_batch

            calls[layer_name] += 1
        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    n = X_sel.shape[0]
    for i in range(0, n, calib_batch):
        end = min(i + calib_batch, n)
        current_batch_w = torch.from_numpy(w_sel[i:end]).float()
        xb = X_sel[i:end]
        _ = forecast_timesfm_point(tfm_model, xb, horizon=horizon)

    for h in hooks:
        h.remove()
    return stats


PAIRS = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long)
PAIR_MASKS = torch.zeros((6, 4), dtype=torch.float32)
for k in range(6):
    i, j = PAIRS[k].tolist()
    PAIR_MASKS[k, i] = 1.0
    PAIR_MASKS[k, j] = 1.0


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
    Gs = (st.Gsig / max(st.Csig, 1e-6)).to(device=device, dtype=dtype)

    masks = PAIR_MASKS.to(device=device, dtype=dtype)
    scores = torch.empty((O, Ggroups, 6), device=device, dtype=dtype)

    for k in range(6):
        mk = masks[k].view(1, 1, 4)
        md = (1.0 - mk)
        Wk = Wg * mk
        Wd = Wg * md

        Tk = torch.einsum("ogc,gcd->ogd", Wk, Gs)
        Ek = (Tk * Wk).sum(dim=2)
        Td = torch.einsum("ogc,gcd->ogd", Wd, Gs)
        Ed = (Td * Wd).sum(dim=2)

        if score_mode == "keep":
            scores[:, :, k] = Ek
        else:
            scores[:, :, k] = Ek / (Ed + eps)

    bestk = torch.argmax(scores, dim=2)
    pair_idx = PAIRS.to(device=device)[bestk]
    keep_mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool)
    keep_mask.scatter_(2, pair_idx, torch.ones_like(pair_idx, dtype=torch.bool))

    if not refit:
        Wnew = torch.where(keep_mask, Wg, torch.zeros_like(Wg))
    else:
        B = torch.einsum("ogc,gcd->ogd", Wg, Gs)
        invs = []
        eye2 = torch.eye(2, device=device, dtype=dtype).view(1, 2, 2)
        for k in range(6):
            i, j = PAIRS[k].tolist()
            Gss = Gs[:, [i, j]][:, :, [i, j]] + ridge * eye2
            invs.append(torch.inverse(Gss))
        invs = torch.stack(invs, dim=0)

        Wnew = torch.zeros_like(Wg)
        g_idx = torch.arange(Ggroups, device=device).view(1, Ggroups).expand(O, Ggroups)
        for k in range(6):
            sel = (bestk == k)
            if not torch.any(sel):
                continue
            i, j = PAIRS[k].tolist()
            b0 = B[:, :, i][sel]
            b1 = B[:, :, j][sel]
            bsel = torch.stack([b0, b1], dim=1)
            gsel = g_idx[sel]
            invsel = invs[k, gsel, :, :]
            u = torch.bmm(invsel, bsel.unsqueeze(2)).squeeze(2)
            Wnew[:, :, i][sel] = u[:, 0]
            Wnew[:, :, j][sel] = u[:, 1]

    W[:, :Cg] = Wnew.view(O, Cg)


def report_effective_sparsity(model: nn.Module, include_regex: str, include_quantile_head: bool):
    pat = re.compile(include_regex) if include_regex else None
    total_model_params = sum(p.numel() for p in model.parameters())

    target_params = 0
    grouped_params = 0
    zero_params = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        nl = name.lower()
        if (not include_quantile_head) and ("output_projection_quantiles" in nl):
            continue
        if pat and not pat.match(name):
            continue

        W = mod.weight.data
        O, C = W.shape
        target_params += W.numel()
        Cg = (C // 4) * 4
        grouped_params += O * Cg
        if Cg > 0:
            zero_params += (W[:, :Cg] == 0).sum().item()

    structured = zero_params / max(grouped_params, 1)
    overall = zero_params / max(total_model_params, 1)

    print("\n================ SPARSITY REPORT ================")
    print(f"Total model params           : {total_model_params:,}")
    print(f"Targeted linear params       : {target_params:,}")
    print(f"2:4 grouped params           : {grouped_params:,}")
    print(f"Zeroed params (2:4 region)   : {zero_params:,}")
    print(f"Structured sparsity (2:4)    : {structured*100:.2f}%")
    print(f"Effective model sparsity     : {overall*100:.2f}%")
    print("=================================================\n")


@torch.no_grad()
def eval_one_series(tfm, series_1d: np.ndarray, train_end: int, context: int, horizon: int,
                    stride_test: int, test_start_window: int,
                    test_windows: int, batch: int, measure_time: bool):
    T = len(series_1d)
    X_test_all, Y_test_all = make_windows_1d(series_1d, train_end, T, context, horizon, stride_test)
    ts = test_start_window
    te = X_test_all.shape[0] if test_windows < 0 else min(ts + test_windows, X_test_all.shape[0])
    X_test, Y_test = X_test_all[ts:te], Y_test_all[ts:te]
    if X_test.shape[0] == 0:
        return float("nan"), float("nan"), 0.0

    if measure_time:
        pred, tsec = timed_forecast(tfm, X_test, horizon, batch)
        mse, mae = mse_mae(pred, Y_test)
        return mse, mae, tsec

    preds = []
    for i in range(0, len(X_test), batch):
        preds.append(forecast_timesfm_point(tfm, X_test[i:i+batch], horizon))
    pred = np.concatenate(preds, axis=0)
    mse, mae = mse_mae(pred, Y_test)
    return mse, mae, 0.0


def choose_indices(rng: np.random.Generator, n: int, k: int, how: str) -> np.ndarray:
    if k < 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return np.arange(k, dtype=np.int64) if how == "first" else rng.choice(n, size=k, replace=False).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--col", type=str, default="", help="Single column name (single-series mode).")
    ap.add_argument("--cols_regex", type=str, default="^MT_\\d+$")

    ap.add_argument("--train_end", type=int, default=49152)
    ap.add_argument("--train_frac", type=float, default=0.7)

    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--stride_test", type=int, default=96)

    ap.add_argument("--calib_cols", type=int, default=32)
    ap.add_argument("--calib_cols_select", type=str, default="random", choices=["first", "random"])
    ap.add_argument("--calib_windows", type=int, default=256)
    ap.add_argument("--calib_select", type=str, default="random", choices=["first", "random", "topk"])

    # NEW: evaluation sampling
    ap.add_argument("--eval_cols", type=int, default=32,
                    help="Multi-series eval columns for macro-average. -1 means ALL.")
    ap.add_argument("--eval_cols_select", type=str, default="random", choices=["first", "random"])

    ap.add_argument("--test_windows", type=int, default=16, help="-1 means ALL (slow).")
    ap.add_argument("--test_start_window", type=int, default=0)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--calib_batch", type=int, default=4)

    ap.add_argument("--include_quantile_head", action="store_true")
    ap.add_argument("--include_regex", type=str, default=".*")
    ap.add_argument("--exclude_regex", type=str, default="")

    ap.add_argument("--sample_rows_per_call", type=int, default=1024)
    ap.add_argument("--max_calls_per_layer", type=int, default=32)

    ap.add_argument("--score_mode", type=str, default="ratio", choices=["ratio", "keep"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-5)

    ap.add_argument("--model_id", type=str, default="google/timesfm-2.5-200m-pytorch")
    ap.add_argument("--error_power", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--measure_time", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    single_mode = bool(args.col)

    if single_mode:
        series = load_series(args.csv, args.col)
        T = len(series)
        train_end = int(T * args.train_frac) if args.train_frac > 0 else args.train_end

        X_train, Y_train = make_windows_1d(series, 0, train_end, args.context, args.horizon, args.stride_train)
        X_pool = X_train[:args.calib_windows]
        Y_pool = Y_train[:args.calib_windows]

        eval_series = [(args.col, series)]
        print(f"[mode] single-series col={args.col}")
    else:
        mat, colnames = load_series_matrix(args.csv, args.cols_regex)
        T, D = mat.shape
        train_end = int(T * args.train_frac) if args.train_frac > 0 else args.train_end
        if train_end <= 0 or train_end >= T:
            raise ValueError(f"train_end must be in (0, T). Got train_end={train_end}, T={T}.")

        calib_js = choose_indices(rng, D, min(D, int(args.calib_cols)), args.calib_cols_select)
        X_pool_list, Y_pool_list = [], []
        for j in calib_js:
            s = mat[:, j]
            X_tr, Y_tr = make_windows_1d(s, 0, train_end, args.context, args.horizon, args.stride_train)
            X_pool_list.append(X_tr[:args.calib_windows])
            Y_pool_list.append(Y_tr[:args.calib_windows])
        X_pool = np.concatenate(X_pool_list, axis=0)
        Y_pool = np.concatenate(Y_pool_list, axis=0)

        eval_js = choose_indices(rng, D, int(args.eval_cols), args.eval_cols_select)
        eval_series = [(colnames[j], mat[:, j]) for j in eval_js]

        print(f"[mode] multi-series cols={D} regex={args.cols_regex}")
        print(f"[calib] cols_used={len(calib_js)} poolN={X_pool.shape[0]}")
        if args.eval_cols < 0 or args.eval_cols >= D:
            print(f"[eval]  FULL macro-avg over ALL {D} cols (slow)")
        else:
            print(f"[eval]  PROXY macro-avg over {len(eval_series)}/{D} cols select={args.eval_cols_select} seed={args.seed}")

    print(f"[split] total_rows={T} train_end={train_end} context={args.context} horizon={args.horizon}")
    print(f"[test]  test_windows={args.test_windows} stride_test={args.stride_test} start={args.test_start_window}")

    gram_budget = args.max_calls_per_layer * args.calib_batch
    eff_K = min(X_pool.shape[0], gram_budget)
    print(f"[calib] eval_batch={args.batch} calib_batch={args.calib_batch} gram_budget={gram_budget} => effective_K={eff_K}")

    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    torch_mod = find_torch_module(tfm)
    targets = select_linears(torch_mod, args.include_quantile_head, args.include_regex, args.exclude_regex)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=max(args.horizon, 256)))

    # baseline eval
    mses_b, maes_b, tb = [], [], []
    for _name, s in eval_series:
        mse, mae, tsec = eval_one_series(
            tfm, s, train_end, args.context, args.horizon,
            args.stride_test, args.test_start_window, args.test_windows,
            args.batch, args.measure_time
        )
        if np.isfinite(mse):
            mses_b.append(mse); maes_b.append(mae)
            if args.measure_time: tb.append(tsec)
    mse_b = float(np.mean(mses_b)) if mses_b else float("nan")
    mae_b = float(np.mean(maes_b)) if maes_b else float("nan")
    if args.measure_time:
        print(f"[baseline] MACRO_AVG over {len(mses_b)}/{len(eval_series)} series: MSE={mse_b:.6f} MAE={mae_b:.6f} | avg_batch_sec={float(np.mean(tb)) if tb else 0.0:.4f}")
    else:
        print(f"[baseline] MACRO_AVG over {len(mses_b)}/{len(eval_series)} series: MSE={mse_b:.6f} MAE={mae_b:.6f}")

    # errors/weights
    need_errors = (args.calib_select == "topk") or (args.error_power != 0.0)
    if need_errors:
        print(f"[calib] computing errors/weights on pool: power={args.error_power} (batched={args.calib_batch}) ...")
        errors, _er, weights = compute_errors_and_weights(tfm, X_pool, Y_pool, args.horizon, args.calib_batch, args.error_power)
        print(f"[calib] weight_stats: min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f} (power={args.error_power})")
    else:
        errors = None
        weights = np.ones((X_pool.shape[0],), dtype=np.float32)
        print(f"[calib] uniform weights (no labels used): power=0, select={args.calib_select}")

    # select K
    K = eff_K
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

    print(f"[calib] collecting grams: select={args.calib_select} pool={X_pool.shape[0]} K={K} (max_calls_per_layer={args.max_calls_per_layer})")
    gram_stats = collect_group_grams(tfm, targets, X_sel, w_sel, args.horizon, args.calib_batch, args.sample_rows_per_call, args.max_calls_per_layer)
    print(f"[calib] collected grams for {len(gram_stats)}/{len(targets)} layers")

    # prune
    print(f"[prune] SNR 2:4: score_mode={args.score_mode}, refit={bool(args.refit)} ridge={args.ridge:g}")
    for name, layer in targets:
        st = gram_stats.get(name, None)
        if st is None or st.Csig <= 0.0:
            continue
        prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), args.ridge)

    report_effective_sparsity(torch_mod, args.include_regex, args.include_quantile_head)

    # pruned eval
    mses_p, maes_p, tp = [], [], []
    for _name, s in eval_series:
        mse, mae, tsec = eval_one_series(
            tfm, s, train_end, args.context, args.horizon,
            args.stride_test, args.test_start_window, args.test_windows,
            args.batch, args.measure_time
        )
        if np.isfinite(mse):
            mses_p.append(mse); maes_p.append(mae)
            if args.measure_time: tp.append(tsec)
    mse_p = float(np.mean(mses_p)) if mses_p else float("nan")
    mae_p = float(np.mean(maes_p)) if maes_p else float("nan")
    if args.measure_time:
        print(f"[snr-2of4-refit] MACRO_AVG over {len(mses_p)}/{len(eval_series)} series: MSE={mse_p:.6f} MAE={mae_p:.6f} | avg_batch_sec={float(np.mean(tp)) if tp else 0.0:.4f}")
    else:
        print(f"[snr-2of4-refit] MACRO_AVG over {len(mses_p)}/{len(eval_series)} series: MSE={mse_p:.6f} MAE={mae_p:.6f}")

    print(f"[delta] ΔMSE={(mse_p - mse_b):+.6f}  ΔMAE={(mae_p - mae_b):+.6f}")


if __name__ == "__main__":
    main()
