#!/usr/bin/env python3
"""
chronos2_nm_prune_baselines.py (fixed Wanda vs SparseGPT)

Key fix:
- Previous "SparseGPT(diag)" was equivalent to WANDA for 2:4 (monotonic square).
- This version implements SparseGPT-style *block-4* inverse-Hessian diagonal proxy:
    score_i ~ w_i^2 / (H^{-1})_ii   where H is 4x4 block Gram per group.

Methods:
- mag        : keep top2 |w|
- wanda      : keep top2 |w| * sqrt(diag(H))
- sparsegpt  : keep top2 w^2 / diag(inv(H))   (block4 OBS proxy)
- none       : no pruning

Also prints both normalized + raw metrics and raw deltas.

Example ETTh2:
python chronos2_nm_prune_baselines.py \
  --csv ETDataset/ETT-small/ETTh2.csv --cols_regex '^OT$' \
  --context 1024 --horizon 96 --train_end 8640 \
  --stride_train 1 --stride_test 96 \
  --calib_windows 1091 --calib_select first \
  --calib_batch 1 --max_calls_per_layer 64 --sample_rows_per_call 2048 \
  --model_id amazon/chronos-2 --device cuda --torch_dtype float16 \
  --batch 4 --measure_time --zscore \
  --method sparsegpt \
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
# Utils / metrics
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
# Chronos loading / forecasting
# -------------------------

def load_chronos_pipeline(model_id: str, device: str, torch_dtype: str):
    dtype = None
    if torch_dtype and torch_dtype.lower() != "none":
        mp = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = mp[torch_dtype.lower()]

    try:
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        return pipe, "chronos2"
    except Exception:
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        return pipe, "chronos"


def _to_point_forecast_from_tensor(t: torch.Tensor, horizon: int) -> np.ndarray:
    t = t.detach().cpu()
    while t.dim() > 1 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.dim() == 1:
        tt = t.flatten()
        if tt.numel() < horizon:
            pad = tt[-1].repeat(horizon - tt.numel())
            tt = torch.cat([tt, pad], dim=0)
        return tt[:horizon].numpy()

    if t.dim() == 2:
        if t.shape[1] != horizon and t.shape[0] == horizon:
            t = t.t()
        mid = t.shape[0] // 2
        return t[mid, :horizon].numpy()

    while t.dim() > 2:
        t = t[0]
    return _to_point_forecast_from_tensor(t, horizon)


@torch.no_grad()
def forecast_batch(pipe, pipe_kind: str, X: np.ndarray, horizon: int, batch_size: int, num_samples: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]

    inputs_list = [X[i].astype(np.float32, copy=False) for i in range(X.shape[0])]

    if pipe_kind == "chronos2":
        outs = pipe.predict(inputs_list, prediction_length=horizon, batch_size=batch_size)
        preds = [_to_point_forecast_from_tensor(t, horizon) for t in outs]
        return np.stack(preds, axis=0)

    try:
        outs = pipe.predict(inputs_list, prediction_length=horizon, num_samples=num_samples, batch_size=batch_size)
    except TypeError:
        outs = pipe.predict(inputs_list, prediction_length=horizon, batch_size=batch_size)

    if isinstance(outs, list):
        preds = [_to_point_forecast_from_tensor(t, horizon) for t in outs]
        return np.stack(preds, axis=0)
    return _to_point_forecast_from_tensor(outs, horizon)[None, :]


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
# Data windowing
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


# -------------------------
# Layer selection + block Hessian stats
# -------------------------

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


@dataclass
class BlockGramStat:
    Gsum: torch.Tensor  # [G,4,4] float32 CPU
    count: int


@torch.no_grad()
def collect_block4_grams(
    pipe,
    pipe_kind: str,
    targets: List[Tuple[str, nn.Linear]],
    X_sel: np.ndarray,
    horizon: int,
    calib_batch: int,
    sample_rows_per_call: int,
    max_calls_per_layer: int,
    num_samples: int,
) -> Dict[str, BlockGramStat]:
    """
    Collect H_g = E[x_g x_g^T] per 4-wide group for each Linear layer input.
    This captures correlation inside each 4-block, which is what makes SparseGPT(block4) differ from Wanda.
    """
    stats: Dict[str, BlockGramStat] = {}
    calls = {name: 0 for name, _ in targets}
    hooks = []

    def make_hook(layer_name: str):
        def pre_hook(_mod, inputs):
            if calls[layer_name] >= max_calls_per_layer:
                return
            x = inputs[0]
            if not torch.is_floating_point(x):
                return

            if x.dim() == 3:
                xf = x.reshape(-1, x.shape[-1])
            else:
                xf = x.reshape(x.shape[0], -1)

            xf = xf.float()

            Cin = xf.shape[-1]
            G = Cin // 4
            Cg = G * 4
            if Cg == 0:
                return

            xf = xf[:, :Cg]

            if xf.shape[0] > sample_rows_per_call:
                idx = torch.randint(0, xf.shape[0], (sample_rows_per_call,), device=xf.device)
                xf = xf.index_select(0, idx)

            N = xf.shape[0]
            xg = xf.reshape(N, G, 4)  # [N,G,4]
            gram = torch.einsum("ngc,ngd->gcd", xg, xg).detach().cpu()  # [G,4,4]

            if layer_name not in stats:
                stats[layer_name] = BlockGramStat(Gsum=gram, count=N)
            else:
                stats[layer_name].Gsum += gram
                stats[layer_name].count += N

            calls[layer_name] += 1
        return pre_hook

    for name, layer in targets:
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))

    n = int(X_sel.shape[0])
    for i in range(0, n, calib_batch):
        end = min(i + calib_batch, n)
        xb = X_sel[i:end]
        _ = forecast_batch(pipe, pipe_kind, xb, horizon, batch_size=calib_batch, num_samples=num_samples)

    for h in hooks:
        h.remove()

    return stats


# -------------------------
# 2:4 pruning baselines
# -------------------------

@torch.no_grad()
def prune_linear_2of4_mag(layer: nn.Linear) -> None:
    W = layer.weight.data
    Cin = int(W.shape[1])
    G = Cin // 4
    Cg = G * 4
    if Cg == 0:
        return

    Wf = W[:, :Cg].float().view(W.shape[0], G, 4)
    scores = Wf.abs()
    top2 = torch.topk(scores, k=2, dim=2, largest=True).indices
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    Wf = Wf * mask.to(Wf.dtype)
    W[:, :Cg] = Wf.reshape(W.shape[0], Cg).to(W.dtype)
    if Cg < Cin:
        W[:, Cg:] = 0


@torch.no_grad()
def prune_linear_2of4_wanda_from_blockH(layer: nn.Linear, H: torch.Tensor, eps: float = 1e-8) -> None:
    """
    H: [G,4,4] float32 CPU. Uses diag(H) only.
    score = |w| * sqrt(diag(H))
    """
    W = layer.weight.data
    Cin = int(W.shape[1])
    G = Cin // 4
    Cg = G * 4
    if Cg == 0:
        return

    H = H.to(device=W.device, dtype=torch.float32)
    diagH = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(0.0)  # [G,4]
    scale = torch.sqrt(diagH + eps)  # [G,4]

    Wf = W[:, :Cg].float().view(W.shape[0], G, 4)
    scores = Wf.abs() * scale.unsqueeze(0)
    top2 = torch.topk(scores, k=2, dim=2, largest=True).indices
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    Wf = Wf * mask.to(Wf.dtype)
    W[:, :Cg] = Wf.reshape(W.shape[0], Cg).to(W.dtype)
    if Cg < Cin:
        W[:, Cg:] = 0


@torch.no_grad()
def prune_linear_2of4_sparsegpt_block4(layer: nn.Linear, H: torch.Tensor, eps: float = 1e-4) -> None:
    """
    SparseGPT-style (OBS proxy) using 4x4 block Hessian:
      saliency_i ~ w_i^2 / (H^{-1})_ii
    Keep top-2 per 4-group by this saliency.

    H: [G,4,4] float32 CPU (block Gram).
    """
    W = layer.weight.data
    Cin = int(W.shape[1])
    G = Cin // 4
    Cg = G * 4
    if Cg == 0:
        return

    H = H.to(dtype=torch.float32)  # keep on CPU for stable inv
    I = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # [1,4,4]
    Hreg = H + eps * I  # [G,4,4]
    invH = torch.linalg.inv(Hreg)  # [G,4,4]
    inv_diag = torch.diagonal(invH, dim1=-2, dim2=-1).clamp_min(1e-12)  # [G,4]

    inv_diag = inv_diag.to(device=W.device, dtype=torch.float32)

    Wf = W[:, :Cg].float().view(W.shape[0], G, 4)
    scores = (Wf * Wf) / inv_diag.unsqueeze(0)  # [O,G,4]
    top2 = torch.topk(scores, k=2, dim=2, largest=True).indices
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    Wf = Wf * mask.to(Wf.dtype)
    W[:, :Cg] = Wf.reshape(W.shape[0], Cg).to(W.dtype)
    if Cg < Cin:
        W[:, Cg:] = 0


# -------------------------
# Main
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
    ap.add_argument("--calib_select", choices=["first", "random"], default="first")
    ap.add_argument("--calib_batch", type=int, default=1)
    ap.add_argument("--max_calls_per_layer", type=int, default=64)
    ap.add_argument("--sample_rows_per_call", type=int, default=2048)

    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--measure_time", action="store_true")

    ap.add_argument("--model_id", default="amazon/chronos-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch_dtype", choices=["None", "float16", "bfloat16", "float32"], default="None")
    ap.add_argument("--num_samples", type=int, default=16)

    ap.add_argument("--include_regex", default=".*")
    ap.add_argument("--exclude_regex", default=r".*(embed|embedding|patch|token|input|pos|lm_head|head|output).*")

    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--method", choices=["none", "mag", "wanda", "sparsegpt"], default="mag")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--sparsegpt_eps", type=float, default=1e-4)
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

    pipe, pipe_kind = load_chronos_pipeline(args.model_id, args.device, args.torch_dtype)
    model = getattr(pipe, "model", pipe)
    targets = find_target_linears(model, args.include_regex, args.exclude_regex)
    print(f"[model] {args.model_id} | pipe_kind={pipe_kind} | targets(nn.Linear)={len(targets)}")

    mse_base_list, mae_base_list, mse_pr_list, mae_pr_list = [], [], [], []
    mse_base_raw_list, mae_base_raw_list, mse_pr_raw_list, mae_pr_raw_list = [], [], [], []

    for ci, col in enumerate(cols):
        series_train = data_n[:train_end, ci]
        series_test = data_n[train_end:, ci]

        X_train, Y_train = build_windows(series_train, args.context, args.horizon, args.stride_train)
        X_test, Y_test = build_windows(series_test, args.context, args.horizon, args.stride_test)

        poolN = min(args.calib_windows, X_train.shape[0]) if args.calib_windows != -1 else X_train.shape[0]
        if args.calib_select == "first":
            pool_idx = np.arange(poolN, dtype=np.int64)
        else:
            pool_idx = np.random.permutation(X_train.shape[0]).astype(np.int64)[:poolN]
        X_pool = X_train[pool_idx]

        print(f"\n[col={col}] train_windows={X_train.shape[0]} calib_pool={poolN} test_windows={X_test.shape[0]}")

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

        if args.method == "none":
            mse_p, mae_p, mse_p_raw, mae_p_raw = mse_b, mae_b, mse_b_raw, mae_b_raw
        else:
            print(f"[calib] collecting block-4 grams for {args.method}: pool={poolN} "
                  f"(max_calls_per_layer={args.max_calls_per_layer}, calib_batch={args.calib_batch})")

            grams = collect_block4_grams(
                pipe=pipe,
                pipe_kind=pipe_kind,
                targets=targets,
                X_sel=X_pool,
                horizon=args.horizon,
                calib_batch=args.calib_batch,
                sample_rows_per_call=args.sample_rows_per_call,
                max_calls_per_layer=args.max_calls_per_layer,
                num_samples=args.num_samples,
            )
            print(f"[calib] grams collected for {len(grams)}/{len(targets)} layers")

            if args.method == "mag":
                print("[prune] MAG 2:4")
                for _, layer in targets:
                    prune_linear_2of4_mag(layer)

            elif args.method == "wanda":
                print("[prune] WANDA 2:4 (uses diag(H))")
                for name, layer in targets:
                    st = grams.get(name, None)
                    if st is None or st.count <= 0:
                        continue
                    H = st.Gsum / float(st.count)
                    prune_linear_2of4_wanda_from_blockH(layer, H, eps=args.eps)

            elif args.method == "sparsegpt":
                print("[prune] SparseGPT(block4 OBS-proxy) 2:4 (uses diag(inv(H)))")
                for name, layer in targets:
                    st = grams.get(name, None)
                    if st is None or st.count <= 0:
                        continue
                    H = st.Gsum / float(st.count)
                    prune_linear_2of4_sparsegpt_block4(layer, H, eps=args.sparsegpt_eps)

            else:
                raise ValueError(f"unknown method {args.method}")

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
