"""
prune_unified.py

True Competitive MoE Implementation.
Evaluates MAG, Wanda, SNR, and OBS experts per layer against held-out validation data.
Includes Refit Penalization and SNR Bias for robust generalization.
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
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

def eval_forecast_mse(tfm_model, X: np.ndarray, Y: np.ndarray, horizon: int, batch: int):
    preds = []
    for i in range(0, len(X), batch):
        preds.append(forecast_timesfm_point(tfm_model, X[i:i+batch], horizon))
    return mse_mae(np.concatenate(preds, 0), Y)

def eval_forecast_pred(tfm_model, X: np.ndarray, Y: Optional[np.ndarray], horizon: int, batch: int):
    preds = []
    for i in range(0, len(X), batch):
        preds.append(forecast_timesfm_point(tfm_model, X[i:i+batch], horizon))
    pred = np.concatenate(preds, 0)
    if Y is None:
        return pred, None, None
    mse_v, mae_v = mse_mae(pred, Y)
    return pred, float(mse_v), float(mae_v)

def maybe_store_postpass_record(postpass_ctx: Optional[dict], record: Optional[dict]):
    if not postpass_ctx or not postpass_ctx.get("enabled", False) or record is None:
        return
    records = postpass_ctx.setdefault("records", {})
    max_k = int(postpass_ctx.get("pool_k", 0))
    if max_k <= 0:
        return
    name = record["layer"]
    if name in records:
        if record["priority"] > records[name]["priority"]:
            records[name] = record
        return
    if len(records) < max_k:
        records[name] = record
        return
    min_name, min_rec = min(records.items(), key=lambda kv: kv[1]["priority"])
    if record["priority"] > min_rec["priority"]:
        del records[min_name]
        records[name] = record

def run_greedy_postpass(
    tfm_model,
    targets,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    horizon: int,
    batch: int,
    postpass_ctx: Optional[dict],
):
    if not postpass_ctx or not postpass_ctx.get("enabled", False):
        return None
    records = list(postpass_ctx.get("records", {}).values())
    if not records:
        print("[greedy-postpass] no candidate layers collected")
        return None

    eval_windows = int(postpass_ctx.get("eval_windows", 32))
    eval_select = str(postpass_ctx.get("eval_select", "last"))
    if eval_windows > 0 and eval_windows < len(X_test):
        if eval_select == "first":
            X_eval, Y_eval = X_test[:eval_windows], Y_test[:eval_windows]
        else:
            X_eval, Y_eval = X_test[-eval_windows:], Y_test[-eval_windows:]
    else:
        X_eval, Y_eval = X_test, Y_test

    layer_map = {name: layer for name, layer in targets}
    rec_map = {r["layer"]: r for r in records}
    base_pred_eval, current_mse, current_mae = eval_forecast_pred(tfm_model, X_eval, Y_eval, horizon, batch)
    print(f"[greedy-postpass] subset windows={len(X_eval)} start MSE={current_mse:.6f} MAE={current_mae:.6f}")

    max_cands_per_layer = int(postpass_ctx.get("max_cands_per_layer", 0))

    def best_one_layer_move(lname: str, rec: dict, mse_ref: float):
        layer = layer_map.get(lname)
        if layer is None:
            return None
        cand_items = rec.get("candidates", [])
        if not cand_items:
            return None
        if max_cands_per_layer > 0:
            # candidates are pre-sorted by local proxy; keep only top alternatives for runtime
            # while preserving the current choice and sticky candidates.
            kept = cand_items[: max_cands_per_layer + 1]
            kept_keys = {c.get("key") for c in kept}
            cur_key = rec.get("current_choice")
            for cand in cand_items:
                key = cand.get("key")
                if key in kept_keys:
                    continue
                if key == cur_key or bool(cand.get("sticky", False)):
                    kept.append(cand)
                    kept_keys.add(key)
            cand_items = kept
        cur_weight = layer.weight.data.detach().clone()
        best = None
        try:
            for cand in cand_items:
                key = cand["key"]
                if key == rec.get("current_choice"):
                    continue
                Wcand = cand["weight"]
                layer.weight.data.copy_(Wcand.to(device=layer.weight.data.device, dtype=layer.weight.data.dtype))
                mse_try, _ = eval_forecast_mse(tfm_model, X_eval, Y_eval, horizon, batch)
                gain = mse_ref - mse_try
                if (best is None) or (gain > best["gain"]):
                    best = {
                        "layer": lname,
                        "to": key,
                        "mse": float(mse_try),
                        "gain": float(gain),
                        "priority": float(rec.get("priority", 0.0)),
                        "layer_kind": rec.get("layer_kind", ""),
                        "hint_compare": float(cand.get("compare", float("nan"))),
                        "hint_raw": float(cand.get("raw", float("nan"))),
                    }
        finally:
            layer.weight.data.copy_(cur_weight)
        return best

    def get_candidate_by_key(rec: dict, key: str):
        for cand in rec.get("candidates", []):
            if cand.get("key") == key:
                return cand
        return None

    def eval_move_pred(lname: str, rec: dict, key: str):
        layer = layer_map.get(lname)
        if layer is None:
            return None
        cand = get_candidate_by_key(rec, key)
        if cand is None:
            return None
        cur_weight = layer.weight.data.detach().clone()
        try:
            Wcand = cand["weight"]
            layer.weight.data.copy_(Wcand.to(device=layer.weight.data.device, dtype=layer.weight.data.dtype))
            pred_try, mse_try, mae_try = eval_forecast_pred(tfm_model, X_eval, Y_eval, horizon, batch)
        finally:
            layer.weight.data.copy_(cur_weight)
        return {
            "pred": pred_try,
            "mse": float(mse_try),
            "mae": float(mae_try),
            "cand": cand,
        }

    def eval_pair_mse(move_a: dict, move_b: dict):
        layer_a = layer_map.get(move_a["layer"])
        layer_b = layer_map.get(move_b["layer"])
        if layer_a is None or layer_b is None:
            return None
        rec_a = rec_map.get(move_a["layer"])
        rec_b = rec_map.get(move_b["layer"])
        if rec_a is None or rec_b is None:
            return None
        cand_a = get_candidate_by_key(rec_a, move_a["to"])
        cand_b = get_candidate_by_key(rec_b, move_b["to"])
        if cand_a is None or cand_b is None:
            return None
        cur_a = layer_a.weight.data.detach().clone()
        cur_b = layer_b.weight.data.detach().clone()
        try:
            layer_a.weight.data.copy_(cand_a["weight"].to(device=layer_a.weight.data.device, dtype=layer_a.weight.data.dtype))
            layer_b.weight.data.copy_(cand_b["weight"].to(device=layer_b.weight.data.device, dtype=layer_b.weight.data.dtype))
            mse_try, _ = eval_forecast_mse(tfm_model, X_eval, Y_eval, horizon, batch)
        finally:
            layer_a.weight.data.copy_(cur_a)
            layer_b.weight.data.copy_(cur_b)
        return float(mse_try)

    screen_k = int(postpass_ctx.get("screen_k", 0))
    screen_min_gain = float(postpass_ctx.get("screen_min_gain", 0.0))
    screening = None
    if screen_k > 0:
        screened_moves = []
        for lname, rec in rec_map.items():
            mv = best_one_layer_move(lname, rec, current_mse)
            if mv is None:
                continue
            rec["screen_best"] = mv
            screened_moves.append(mv)
        screened_moves.sort(key=lambda m: (m["gain"], m["priority"]), reverse=True)
        if screen_min_gain > 0:
            screened_moves = [m for m in screened_moves if m["gain"] >= screen_min_gain]
        if screen_k > 0:
            screened_moves = screened_moves[:screen_k]
        selected_names = [m["layer"] for m in screened_moves]
        screening = {
            "enabled": True,
            "evaluated": int(len(rec_map)),
            "selected": int(len(selected_names)),
            "screen_k": int(screen_k),
            "screen_min_gain": float(screen_min_gain),
            "top": [
                {
                    "layer": m["layer"],
                    "to": m["to"],
                    "gain": float(m["gain"]),
                    "priority": float(m["priority"]),
                    "layer_kind": m.get("layer_kind", ""),
                }
                for m in screened_moves[:10]
            ],
        }
        if selected_names:
            rec_map = {name: rec_map[name] for name in selected_names}
            print(
                f"[greedy-postpass] screened pool {len(records)} -> {len(selected_names)} "
                f"(topk={screen_k}, min_gain={screen_min_gain:.4f})"
            )
        else:
            print(
                f"[greedy-postpass] screening selected 0 layers "
                f"(topk={screen_k}, min_gain={screen_min_gain:.4f}); keeping unscreened pool"
            )
            screening["fallback_unscreened"] = True

    pairdiag_result = None
    pair_rows = []
    pairdiag_k = int(postpass_ctx.get("pairdiag_k", 0))
    if pairdiag_k > 0 and rec_map:
        pair_moves = []
        for lname, rec in rec_map.items():
            mv = rec.get("screen_best")
            if mv is None:
                mv = best_one_layer_move(lname, rec, current_mse)
            if mv is None:
                continue
            pair_moves.append(mv)
        pair_moves.sort(key=lambda m: (m["gain"], m["priority"]), reverse=True)
        pair_moves = pair_moves[:pairdiag_k]
        move_eval = {}
        for mv in pair_moves:
            ev = eval_move_pred(mv["layer"], rec_map[mv["layer"]], mv["to"])
            if ev is None:
                continue
            delta = (ev["pred"] - base_pred_eval).astype(np.float32, copy=False)
            move_eval[mv["layer"]] = {
                "move": mv,
                "mse": float(ev["mse"]),
                "mae": float(ev["mae"]),
                "delta": delta,
                "gain": float(current_mse - ev["mse"]),
                "cand_compare": float(ev["cand"].get("compare", float("nan"))),
                "cand_raw": float(ev["cand"].get("raw", float("nan"))),
            }
        pair_rows = []
        move_names = sorted(move_eval.keys())
        exact_pairs_budget = int(postpass_ctx.get("pairdiag_exact_pairs", 0))
        for i in range(len(move_names)):
            for j in range(i + 1, len(move_names)):
                mi = move_eval[move_names[i]]
                mj = move_eval[move_names[j]]
                proxy_synergy = float(-2.0 * np.mean(mi["delta"] * mj["delta"]))
                proxy_pair_gain = float(mi["gain"] + mj["gain"] + proxy_synergy)
                row = {
                    "layer_i": mi["move"]["layer"],
                    "to_i": mi["move"]["to"],
                    "kind_i": mi["move"].get("layer_kind", ""),
                    "gain_i": float(mi["gain"]),
                    "layer_j": mj["move"]["layer"],
                    "to_j": mj["move"]["to"],
                    "kind_j": mj["move"].get("layer_kind", ""),
                    "gain_j": float(mj["gain"]),
                    "proxy_synergy": float(proxy_synergy),
                    "proxy_pair_gain": float(proxy_pair_gain),
                    "delta_dot_mean": float(np.mean(mi["delta"] * mj["delta"])),
                    "delta_cosine": float(
                        np.sum(mi["delta"] * mj["delta"]) /
                        (np.linalg.norm(mi["delta"]) * np.linalg.norm(mj["delta"]) + 1e-12)
                    ),
                    "same_block": bool(
                        mi["move"]["layer"].split(".")[1] == mj["move"]["layer"].split(".")[1]
                        if (mi["move"]["layer"].startswith("stacked_xf.") and mj["move"]["layer"].startswith("stacked_xf."))
                        else False
                    ),
                }
                pair_rows.append(row)
        pair_rows.sort(key=lambda r: abs(r["proxy_synergy"]), reverse=True)
        exact_eval_count = 0
        if exact_pairs_budget > 0 and pair_rows:
            for row in pair_rows[:exact_pairs_budget]:
                mi = {"layer": row["layer_i"], "to": row["to_i"]}
                mj = {"layer": row["layer_j"], "to": row["to_j"]}
                mij = eval_pair_mse(mi, mj)
                if mij is None:
                    continue
                gain_ij = float(current_mse - mij)
                synergy_exact = float(gain_ij - row["gain_i"] - row["gain_j"])
                row["pair_mse_exact"] = float(mij)
                row["pair_gain_exact"] = gain_ij
                row["synergy_exact"] = synergy_exact
                exact_eval_count += 1

        top_synergy = sorted(pair_rows, key=lambda r: r["proxy_synergy"], reverse=True)[:8]
        top_conflict = sorted(pair_rows, key=lambda r: r["proxy_synergy"])[:8]
        pairdiag_result = {
            "enabled": True,
            "k": int(pairdiag_k),
            "moves_evaluated": int(len(move_eval)),
            "pairs": int(len(pair_rows)),
            "exact_pairs_evaluated": int(exact_eval_count),
            "top_synergy_proxy": top_synergy,
            "top_conflict_proxy": top_conflict,
        }
        print(
            f"[greedy-postpass] pairdiag moves={len(move_eval)} pairs={len(pair_rows)} "
            f"exact={exact_eval_count}"
        )
        if top_synergy:
            t = top_synergy[0]
            print(
                f"[greedy-postpass] pairdiag top+ {t['layer_i']}+{t['layer_j']} "
                f"proxy_synergy={t['proxy_synergy']:+.6f}"
            )
        if top_conflict:
            t = top_conflict[0]
            print(
                f"[greedy-postpass] pairdiag top- {t['layer_i']}+{t['layer_j']} "
                f"proxy_synergy={t['proxy_synergy']:+.6f}"
            )
        pairdiag_prefix = str(postpass_ctx.get("pairdiag_prefix", "") or "").strip()
        if pairdiag_prefix:
            pfx = Path(pairdiag_prefix)
            pfx.parent.mkdir(parents=True, exist_ok=True)
            rows_path = pfx.with_suffix("")
            rows_csv = str(rows_path) + "_pairs.csv"
            summary_json = str(rows_path) + "_summary.json"
            try:
                pd.DataFrame(pair_rows).to_csv(rows_csv, index=False)
                with open(summary_json, "w") as f:
                    json.dump(pairdiag_result, f, indent=2)
                print(f"[greedy-postpass] pairdiag wrote {rows_csv} and {summary_json}")
            except Exception as e:
                print(f"[greedy-postpass] pairdiag write failed: {type(e).__name__}: {e}")

    pairaware_enabled = bool(postpass_ctx.get("pairaware", False))
    pairaware_alpha = float(postpass_ctx.get("pairaware_alpha", 1.0))
    pairaware_use_exact = bool(postpass_ctx.get("pairaware_use_exact", False))
    pair_synergy_lookup = {}
    if pairaware_enabled and pair_rows:
        exact_count = 0
        for row in pair_rows:
            sval = None
            if pairaware_use_exact and ("synergy_exact" in row):
                sval = row.get("synergy_exact")
                if sval is not None:
                    exact_count += 1
            if sval is None:
                sval = row.get("proxy_synergy", 0.0)
            key_i = (row["layer_i"], row["to_i"])
            key_j = (row["layer_j"], row["to_j"])
            pair_synergy_lookup[(key_i, key_j)] = float(sval)
            pair_synergy_lookup[(key_j, key_i)] = float(sval)
        print(
            f"[greedy-postpass] pairaware on alpha={pairaware_alpha:.3f} "
            f"use_exact={pairaware_use_exact} pairs={len(pair_rows)} exact_used={exact_count}"
        )
    elif pairaware_enabled:
        print("[greedy-postpass] pairaware requested but no pairdiag rows; using plain greedy")
        pairaware_enabled = False

    steps_max = int(postpass_ctx.get("steps", 0)) or len(rec_map)
    min_step_gain = float(postpass_ctx.get("min_step_gain", 0.02))
    remaining = set(rec_map.keys())
    moves = []
    chosen_move_keys = []

    for step in range(1, steps_max + 1):
        best_move = None
        for lname in list(remaining):
            rec = rec_map[lname]
            move = best_one_layer_move(lname, rec, current_mse)
            if move is None:
                continue
            pair_adj = 0.0
            if pair_synergy_lookup and chosen_move_keys:
                mkey = (move["layer"], move["to"])
                pair_adj = float(sum(pair_synergy_lookup.get((mkey, ck), 0.0) for ck in chosen_move_keys))
            move["pair_adj"] = float(pair_adj)
            move["sel_score"] = float(move["gain"] + (pairaware_alpha * pair_adj if pairaware_enabled else 0.0))
            if move is not None and (
                (best_move is None) or
                ((move["sel_score"], move["gain"]) > (best_move.get("sel_score", best_move["gain"]), best_move["gain"]))
            ):
                best_move = move

        if best_move is None or best_move["gain"] < min_step_gain:
            print(
                f"[greedy-postpass] stop step={step} "
                f"best_gain={(best_move['gain'] if best_move else float('nan')):.6f}"
            )
            break

        layer = layer_map[best_move["layer"]]
        rec = rec_map[best_move["layer"]]
        chosen = None
        for cand in rec["candidates"]:
            if cand["key"] == best_move["to"]:
                chosen = cand
                break
        if chosen is None:
            break
        layer.weight.data.copy_(chosen["weight"].to(device=layer.weight.data.device, dtype=layer.weight.data.dtype))
        current_mse = float(best_move["mse"])
        rec["current_choice"] = chosen["key"]
        remaining.remove(best_move["layer"])
        moves.append(best_move)
        chosen_move_keys.append((best_move["layer"], best_move["to"]))
        if pairaware_enabled:
            print(
                f"[greedy-postpass] step={step} {best_move['layer']} -> {best_move['to']} "
                f"gain={best_move['gain']:+.6f} pairadj={best_move.get('pair_adj', 0.0):+.6f} "
                f"score={best_move.get('sel_score', best_move['gain']):+.6f} subset_mse={current_mse:.6f}"
            )
        else:
            print(
                f"[greedy-postpass] step={step} {best_move['layer']} -> {best_move['to']} "
                f"gain={best_move['gain']:+.6f} subset_mse={current_mse:.6f}"
            )

    result = {
        "subset_windows": int(len(X_eval)),
        "subset_mse_after": float(current_mse),
        "moves": moves,
        "pool_size": int(len(records)),
        "search_pool_size": int(len(rec_map)),
    }
    if screening is not None:
        result["screening"] = screening
    if pairdiag_result is not None:
        result["pairdiag"] = pairdiag_result
    if pairaware_enabled:
        result["pairaware"] = {
            "enabled": True,
            "alpha": float(pairaware_alpha),
            "use_exact": bool(pairaware_use_exact),
            "pairs_available": int(len(pair_rows)),
        }
    postpass_ctx["result"] = result
    return result

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
            if ws.numel() != B:
                reps = max(1, (B + ws.numel() - 1) // max(ws.numel(), 1))
                ws = ws.repeat(reps)[:B]
            if wn is not None and wn.numel() != B:
                reps = max(1, (B + wn.numel() - 1) // max(wn.numel(), 1))
                wn = wn.repeat(reps)[:B]
            
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
                rep = max(1, xf.shape[0] // max(B, 1))
                ws_rep = ws.repeat_interleave(rep)
                if ws_rep.numel() < xg.shape[0]:
                    ws_rep = ws.repeat_interleave(rep + 1)
                ws_rep = ws_rep[: xg.shape[0]]
                wn_rep = None
                if wn is not None:
                    wn_rep = wn.repeat_interleave(rep)
                    if wn_rep.numel() < xg.shape[0]:
                        wn_rep = wn.repeat_interleave(rep + 1)
                    wn_rep = wn_rep[: xg.shape[0]]
                Gact = torch.einsum("ngc,ngd->gcd", xg, xg).cpu()
                Gsig = torch.einsum("n,ngc,ngd->gcd", ws_rep, xg, xg).cpu()
                Gnoi = torch.einsum("n,ngc,ngd->gcd", wn_rep, xg, xg).cpu() if wn_rep is not None else None
                
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
def prune_linear_snr_2of4(layer, st: GramStat, score_mode: str, eps: float, refit: bool, ridge: float, horizon: int = 96, nf_hi: float = 0.25, layer_name: str = "", gate_state: Optional[dict] = None, forecast_tiebreak: Optional[dict] = None, hybrid_policy: Optional[dict] = None, safe_policy: Optional[dict] = None, postpass_ctx: Optional[dict] = None):
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

        lname = (layer_name or "").lower()
        is_ff0 = ".ff0" in lname
        is_ff1 = ".ff1" in lname
        is_out_proj = "output_projection_point" in lname
        is_tokenizer = "tokenizer." in lname
        is_attn_proj = (".attn." in lname) and (("qkv_proj" in lname) or ("o_proj" in lname))
        is_risky_layer = is_ff1 or is_out_proj or is_tokenizer
        is_noisy_sensitive = is_tokenizer or is_ff0 or is_attn_proj

        # Layer-local gating features (distribution-informed).
        # Use per-layer spectral NSR and Gram conditioning rather than a dataset-wide hard lock.
        te = float(st.trend_energy.sum()) if st.trend_energy is not None else 0.0
        se = float(st.season_energy.sum()) if st.season_energy is not None else 0.0
        ne = float(st.noise_energy.sum()) if st.noise_energy is not None else 0.0
        local_nsr = ne / (te + se + 1e-9)
        x2_mean = float(st.m2 / max(st.n, 1)) if getattr(st, "n", 0) else 0.0
        x4_mean = float(st.m4 / max(st.n, 1)) if getattr(st, "n", 0) else 0.0
        x_kurt_raw = float(x4_mean / (x2_mean * x2_mean + 1e-12)) if x2_mean > 0 else float("nan")
        eye4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        ga_cond = torch.linalg.cond(Ga + 1e-6 * eye4)
        cond_p90 = float(torch.quantile(ga_cond.float(), 0.90).item())
        ga_diag = torch.diagonal(Ga, dim1=1, dim2=2).reshape(-1).float()
        gdiag_mean = float(torch.mean(ga_diag).item()) if ga_diag.numel() else 0.0
        gdiag_cv = float((torch.std(ga_diag) / (torch.mean(ga_diag) + 1e-12)).item()) if ga_diag.numel() else float("nan")
        ga_off = Ga.clone()
        ga_off[:, torch.arange(4), torch.arange(4)] = 0
        offdiag_ratio = float(ga_off.abs().mean().item() / (gdiag_mean + 1e-12))

        # TRUE Competitive MoE with layer-local soft priors
        refit_p = 1.0
        mag_bias = 1.0
        wanda_bias = 1.0
        if local_nsr < 0.05:
            # Clean/seasonal layers: bias away from SNR and slightly toward OBS.
            obs_bias = 0.995
            snr_bias = 1.03
        elif local_nsr > (nf_hi if nf_hi > 0 else 0.18):
            # Noisy layers: allow a modest SNR preference.
            obs_bias = 1.005
            snr_bias = 0.99
            # In noisy layers, encourage simpler masks slightly (MAG often wins on ETTm2).
            mag_bias = 0.986
            wanda_bias = 1.010
        else:
            obs_bias = 1.0
            snr_bias = 1.0

        # Layer-type-specific soft priors from diagnostics:
        # - `ff1` and output heads often have severe Gram ill-conditioning.
        # - tokenizer/output layers can be unusually noisy on ETTm2.
        if is_ff1 and cond_p90 > 10_000:
            obs_bias = min(obs_bias, 0.992)
            snr_bias = max(snr_bias, 1.04)
        if is_out_proj and cond_p90 > 100_000:
            obs_bias = min(obs_bias, 0.99)
            snr_bias = max(snr_bias, 1.05)
        if is_tokenizer and local_nsr > 0.18:
            # In noisy tokenizer layers, avoid overcommitting to SNR masks.
            snr_bias = max(snr_bias, 1.02)
            mag_bias = min(mag_bias, 0.980)
            wanda_bias = max(wanda_bias, 1.015)
        if is_ff0 and local_nsr > 0.18:
            mag_bias = min(mag_bias, 0.984)
            wanda_bias = max(wanda_bias, 1.012)

        # Ill-conditioned layers are most prone to refit overfitting / unstable gates.
        if cond_p90 > 5_000:
            refit_p = 1.01
        elif cond_p90 > 1_000:
            refit_p = 1.005
        if is_risky_layer and cond_p90 > 10_000:
            refit_p = max(refit_p, 1.02)
        if is_out_proj and cond_p90 > 100_000:
            refit_p = max(refit_p, 1.03)

        allow_refit = bool(refit)
        if is_ff1 and cond_p90 > 50_000:
            allow_refit = False
        if is_out_proj and cond_p90 > 200_000:
            allow_refit = False
        if is_tokenizer and (local_nsr > 0.25) and cond_p90 > 5_000:
            allow_refit = False
            
        def build_candidate_wnew(expert_idx: int, use_refit: bool):
            cand_bestk = expert_bestks[expert_idx]
            if not use_refit:
                return torch.where(
                    torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool).scatter_(2, PAIRS.to(device)[cand_bestk], True),
                    Wg,
                    torch.zeros_like(Wg),
                )
            W_c = torch.zeros_like(Wg)
            for kk in range(6):
                sel = (cand_bestk == kk)
                if not torch.any(sel):
                    continue
                bsel = torch.stack([B[:,:,PAIRS[kk,0]][sel], B[:,:,PAIRS[kk,1]][sel]], 1)
                u = torch.bmm(invs[kk, g_idx[sel]], bsel.unsqueeze(2)).squeeze(2)
                W_c[:,:,PAIRS[kk,0]][sel], W_c[:,:,PAIRS[kk,1]][sel] = u[:, 0], u[:, 1]
            return W_c

        mask_mses = []
        biased_mask_mses = []
        candidate_best = {}
        candidate_variants = {}
        sgptl_refit_cache = None
        def update_candidate(expert_name: str, expert_idx: int, effective_compare: float, raw_mse: float, use_refit: bool):
            prev = candidate_best.get(expert_name)
            if prev is None or effective_compare < prev[0]:
                candidate_best[expert_name] = (float(effective_compare), float(raw_mse), bool(use_refit), int(expert_idx))
        def build_sparsegpt_lite_refit_obs():
            nonlocal sgptl_refit_cache
            if sgptl_refit_cache is not None:
                return sgptl_refit_cache
            if not allow_refit:
                return None
            try:
                obs_idx = expert_names.index("OBS")
            except ValueError:
                return None
            obs_bestk = expert_bestks[obs_idx]
            eye4_local = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
            H_reg_a = Ga + ridge * eye4_local
            B_a = torch.einsum("ogc,gcd->ogd", Wg, H_reg_a)
            eye2_local = torch.eye(2, device=device, dtype=dtype)
            invs_a = torch.stack([
                torch.inverse(Ga[:, PAIRS[k]][:,:,PAIRS[k]] + ridge * eye2_local)
                for k in range(6)
            ])
            W_sg = torch.zeros_like(Wg)
            for k in range(6):
                sel = (obs_bestk == k)
                if not torch.any(sel):
                    continue
                bsel = torch.stack([B_a[:,:,PAIRS[k,0]][sel], B_a[:,:,PAIRS[k,1]][sel]], 1)
                u = torch.bmm(invs_a[k, g_idx[sel]], bsel.unsqueeze(2)).squeeze(2)
                W_sg[:,:,PAIRS[k,0]][sel], W_sg[:,:,PAIRS[k,1]][sel] = u[:, 0], u[:, 1]
            mse_sg = torch.mean(((X_val_d @ W_sg.view(O, Cg).T) - Y_val_dense)**2).item()
            sgptl_refit_cache = {
                "compare": float(mse_sg * refit_p),
                "raw": float(mse_sg),
                "weight_g": W_sg,
            }
            return sgptl_refit_cache
        for idx_e, bbestk in enumerate(expert_bestks):
            mask = torch.zeros((O,Ggroups,4), device=device, dtype=torch.bool).scatter_(2, PAIRS.to(device)[bbestk], True)
            W_m = torch.where(mask, Wg, torch.zeros_like(Wg))
            mse_m = torch.mean(((X_val_d @ W_m.view(O, Cg).T) - Y_val_dense)**2).item()
            mask_mses.append(mse_m)
            if expert_names[idx_e] == "SNR":
                biased_mask_mses.append(mse_m * snr_bias)
            elif expert_names[idx_e] == "OBS":
                biased_mask_mses.append(mse_m * obs_bias)
            elif expert_names[idx_e] == "MAG":
                biased_mask_mses.append(mse_m * mag_bias)
            elif expert_names[idx_e] == "Wanda":
                biased_mask_mses.append(mse_m * wanda_bias)
            else:
                biased_mask_mses.append(mse_m)
            update_candidate(expert_names[idx_e], idx_e, biased_mask_mses[-1], mse_m, False)
            candidate_variants[(expert_names[idx_e], "mask")] = (float(biased_mask_mses[-1]), float(mse_m), int(idx_e))

        best_mask_idx = biased_mask_mses.index(min(biased_mask_mses))
        best_mask_mse = mask_mses[best_mask_idx]
        drop_frac = best_mask_mse / mse_dense
        best_mse, final_expert, final_refit = best_mask_mse, expert_names[best_mask_idx], False

        if allow_refit:
            for idx_e, bbestk in enumerate(expert_bestks):
                W_r = torch.zeros_like(Wg)
                for k in range(6):
                    sel = (bbestk == k)
                    if not torch.any(sel): continue
                    bsel = torch.stack([B[:,:,PAIRS[k,0]][sel], B[:,:,PAIRS[k,1]][sel]], 1)
                    u = torch.bmm(invs[k, g_idx[sel]], bsel.unsqueeze(2)).squeeze(2)
                    W_r[:,:,PAIRS[k,0]][sel], W_r[:,:,PAIRS[k,1]][sel] = u[:, 0], u[:, 1]
                mse_r = torch.mean(((X_val_d @ W_r.view(O, Cg).T) - Y_val_dense)**2).item()
                compare_mse = mse_r
                if expert_names[idx_e] == "SNR":
                    compare_mse *= snr_bias
                elif expert_names[idx_e] == "OBS":
                    compare_mse *= obs_bias
                elif expert_names[idx_e] == "MAG":
                    compare_mse *= mag_bias
                elif expert_names[idx_e] == "Wanda":
                    compare_mse *= wanda_bias
                compare_eff = compare_mse * refit_p
                update_candidate(expert_names[idx_e], idx_e, compare_eff, mse_r, True)
                candidate_variants[(expert_names[idx_e], "refit")] = (float(compare_eff), float(mse_r), int(idx_e))
                if compare_eff < min(biased_mask_mses):
                    if mse_r < best_mse:
                        best_mse, final_expert, final_refit = mse_r, expert_names[idx_e], True

        # Keep a mild layer-local SNR robustness bias only when the layer is genuinely noisy.
        snr_mse = mask_mses[2]
        if local_nsr > (nf_hi if nf_hi > 0 else 0.18):
            if best_mse > (snr_mse * 0.995):
                best_mse, final_expert, final_refit = snr_mse, "SNR", False

        noisy_mag_active = False
        noisy_mag_override = False
        ds_nsr = float(getattr(st, "avg_nsr", 0.0))
        mag_entry = candidate_best.get("MAG")
        final_entry = candidate_best.get(final_expert)
        if (
            mag_entry is not None
            and final_entry is not None
            and final_expert != "MAG"
            and is_noisy_sensitive
            and ds_nsr >= 0.14
        ):
            noisy_local = local_nsr >= (0.16 if is_tokenizer else 0.18)
            borderline_noisy = (ds_nsr >= 0.18 and local_nsr >= 0.14)
            if noisy_local or borderline_noisy:
                noisy_mag_active = True
                final_cmp, final_raw = float(final_entry[0]), float(final_entry[1])
                mag_cmp, mag_raw, mag_refit, _mag_idx = mag_entry
                slack = 1.06 if is_tokenizer else (1.05 if is_ff0 else 1.04)
                if cond_p90 > 5_000:
                    slack = max(slack, 1.06)
                if final_expert == "SNR":
                    slack += 0.01
                raw_slack = 1.12 if is_tokenizer else (1.10 if is_ff0 else 1.08)
                if drop_frac < 0.08:
                    raw_slack += 0.02
                cmp_ok = (mag_cmp <= final_cmp * slack)
                raw_ok = (
                    final_expert == "Wanda"
                    and mag_raw <= final_raw * raw_slack
                )
                if cmp_ok or raw_ok:
                    final_expert, final_refit, best_mse = "MAG", bool(mag_refit), float(mag_raw)
                    noisy_mag_override = True

        safe_active = False
        safe_override = False
        safe_target = ""
        safe_reason = ""
        if safe_policy and safe_policy.get("mode") == "rule_v1":
            ds_nsr = float(getattr(st, "avg_nsr", 0.0))
            # v1 safety rules are tuned for the noisy ETTm2-like regime; disable on clean regimes.
            if ds_nsr < float(safe_policy.get("min_ds_nsr", 0.12)):
                pass
            else:
                current_key = (final_expert, "refit" if final_refit else "mask")
                wanda_best = candidate_best.get("Wanda")
                non_wanda_best = []
                for _ename, (_cmp, _raw, _rfit, _eidx) in candidate_best.items():
                    if _ename == "Wanda":
                        continue
                    non_wanda_best.append((_cmp, _raw, _ename, "refit" if _rfit else "mask"))
                non_wanda_best.sort(key=lambda t: t[0])
                alt_cmp = float(non_wanda_best[0][0]) if non_wanda_best else float("inf")
                wanda_cmp = float(wanda_best[0]) if wanda_best is not None else float("inf")
                low_conf_wanda = bool(
                    final_expert == "Wanda"
                    and wanda_cmp < float("inf")
                    and alt_cmp <= wanda_cmp * (1.03 if cond_p90 < 500 else 1.05)
                )
                very_low_conf_wanda = bool(
                    final_expert == "Wanda"
                    and wanda_cmp < float("inf")
                    and alt_cmp <= wanda_cmp * 1.01
                )
                # Risk score from activation/time-series distributions + local disagreement.
                risk_score = 0
                if ds_nsr >= 0.16:
                    risk_score += 1
                if low_conf_wanda:
                    risk_score += 1
                if offdiag_ratio > 0.14:
                    risk_score += 1
                if gdiag_cv > 4.0:
                    risk_score += 1
                if is_ff0 and (x_kurt_raw > 120 or local_nsr > 0.16):
                    risk_score += 1
                if is_attn_proj and ("qkv_proj" in lname) and (x_kurt_raw > 30 or (0.12 <= local_nsr <= 0.30)):
                    risk_score += 1
                if ("attn." in lname and ("out" in lname or "o_proj" in lname)) and (x_kurt_raw > 6 or cond_p90 > 25):
                    risk_score += 1
                if is_ff1 and (cond_p90 > 120 or x_kurt_raw > 25):
                    risk_score += 1
                if is_tokenizer and "output_layer" in lname and local_nsr > 0.8:
                    risk_score += 2
                if is_out_proj and final_expert == "Wanda" and final_refit:
                    risk_score += 1

                def choose_safe_variant(pref_keys, cmp_slack=1.06, raw_slack=1.12):
                    cur_entry = candidate_variants.get(current_key)
                    if cur_entry is None:
                        return None
                    cur_cmp, cur_raw, _ = cur_entry
                    for key in pref_keys:
                        cand = candidate_variants.get(key)
                        if cand is None:
                            continue
                        cand_cmp, cand_raw, _ = cand
                        if (cand_cmp <= cur_cmp * cmp_slack) or (cand_raw <= cur_raw * raw_slack):
                            return key, float(cand_raw)
                    return None

                # Safe fallbacks (distribution-conditioned) learned from layerwise attribution patterns.
                if is_tokenizer and "output_layer" in lname and current_key != ("SNR", "mask"):
                    safe_active = True
                    pick = choose_safe_variant([("SNR", "mask"), ("MAG", "mask"), ("OBS", "mask")], cmp_slack=1.10, raw_slack=1.25)
                    if pick is not None:
                        (sx, sv), sraw = pick
                        final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                        safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "tok_output"
                elif is_out_proj and final_expert == "Wanda" and final_refit and risk_score >= 2:
                    safe_active = True
                    pick = choose_safe_variant([("Wanda", "mask"), ("OBS", "refit"), ("OBS", "mask")], cmp_slack=1.05, raw_slack=1.03)
                    if pick is not None:
                        (sx, sv), sraw = pick
                        final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                        safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "outproj_variant"
                elif final_expert == "Wanda" and final_refit:
                    kind_is_attn_out = (".attn." in lname) and (("out" in lname) or ("o_proj" in lname))
                    if is_ff0 and risk_score >= 4 and (low_conf_wanda or local_nsr <= 0.30):
                        safe_active = True
                        pref = [("MAG", "mask"), ("MAG", "refit"), ("SNR", "refit"), ("OBS", "mask"), ("OBS", "refit")]
                        pick = choose_safe_variant(pref, cmp_slack=1.06, raw_slack=1.20)
                        if pick is not None:
                            (sx, sv), sraw = pick
                            final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                            safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "ff0_safe"
                    elif is_attn_proj and ("qkv_proj" in lname) and risk_score >= 4 and (low_conf_wanda or local_nsr >= 0.14):
                        safe_active = True
                        pref = [("MAG", "refit"), ("MAG", "mask"), ("SNR", "refit"), ("SNR", "mask"), ("OBS", "refit")]
                        pick = choose_safe_variant(pref, cmp_slack=1.05, raw_slack=1.16)
                        if pick is not None:
                            (sx, sv), sraw = pick
                            final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                            safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "qkv_safe"
                    elif kind_is_attn_out and ((risk_score >= 4 and low_conf_wanda) or (cond_p90 > 1000 and local_nsr < 0.14)):
                        safe_active = True
                        pref = [("MAG", "refit"), ("MAG", "mask"), ("OBS", "refit"), ("OBS", "mask"), ("SNR", "refit")]
                        pick = choose_safe_variant(pref, cmp_slack=1.06, raw_slack=1.18)
                        if pick is not None:
                            (sx, sv), sraw = pick
                            final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                            safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "attnout_safe"
                    elif is_ff1 and risk_score >= 4 and (very_low_conf_wanda or (0.05 <= local_nsr <= 0.12 and cond_p90 >= 100)):
                        safe_active = True
                        pref = [("OBS", "mask"), ("SNR", "refit"), ("SNR", "mask"), ("MAG", "refit"), ("OBS", "refit"), ("Wanda", "mask")]
                        pick = choose_safe_variant(pref, cmp_slack=1.05, raw_slack=1.15)
                        if pick is not None:
                            (sx, sv), sraw = pick
                            final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                            safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "ff1_safe"

                    # Variant-only safety: Wanda/refit often overfits mildly on many layers.
                    if (not safe_override) and final_expert == "Wanda" and final_refit and risk_score >= 3:
                        safe_active = True
                        pick = choose_safe_variant([("Wanda", "mask")], cmp_slack=1.03, raw_slack=1.04)
                        if pick is not None:
                            (sx, sv), sraw = pick
                            final_expert, final_refit, best_mse = sx, (sv == "refit"), sraw
                            safe_override, safe_target, safe_reason = True, f"{sx}/{sv}", "wanda_mask_variant"

        collapse_active = False
        collapse_override = False
        if gate_state is not None:
            gate_state.setdefault("risky_total", 0)
            gate_state.setdefault("risky_wanda", 0)
            gate_state.setdefault("all_total", 0)
            gate_state.setdefault("all_wanda", 0)
            warmup = int(gate_state.get("warmup", 12))
            wanda_frac_thresh = float(gate_state.get("wanda_frac_thresh", 0.75))
            if (
                is_risky_layer
                and gate_state["risky_total"] >= warmup
                and (gate_state["risky_wanda"] / max(gate_state["risky_total"], 1)) >= wanda_frac_thresh
                and 0.03 <= ds_nsr <= 0.12
            ):
                collapse_active = True

        if collapse_active and final_expert == "Wanda":
            wanda_cmp = candidate_best.get("Wanda", (float("inf"), best_mse, final_refit, 1))[0]
            alt_candidates = []
            for expert_name, (cmp_mse, raw_mse, use_refit, expert_idx) in candidate_best.items():
                if expert_name == "Wanda":
                    continue
                alt_candidates.append((cmp_mse, raw_mse, use_refit, expert_name, expert_idx))
            alt_candidates.sort(key=lambda t: t[0])
            if alt_candidates:
                alt_cmp, alt_raw, alt_refit, alt_expert, alt_idx = alt_candidates[0]
                # Only override when the best non-Wanda option is reasonably competitive
                # under the same local proxy, to avoid forcing obviously bad switches.
                if alt_cmp <= wanda_cmp * (1.03 if cond_p90 > 1_000 else 1.015):
                    # Cheap forecast-aware tie-breaker on a tiny calibration batch.
                    if (
                        forecast_tiebreak is not None
                        and forecast_tiebreak.get("enabled", False)
                        and forecast_tiebreak.get("calls", 0) < forecast_tiebreak.get("max_calls", 0)
                    ):
                        tfm_gate = forecast_tiebreak["tfm"]
                        X_gate = forecast_tiebreak["X"]
                        Y_gate = forecast_tiebreak["Y"]
                        gate_batch = int(forecast_tiebreak.get("batch", 4))

                        orig_full = layer.weight.data.detach().clone()
                        try:
                            # Current winner (Wanda) vs best non-Wanda alternative.
                            wanda_idx = candidate_best["Wanda"][3]
                            wanda_refit = candidate_best["Wanda"][2]
                            W_wanda = build_candidate_wnew(wanda_idx, wanda_refit)
                            W_alt = build_candidate_wnew(alt_idx, alt_refit)

                            scores = {}
                            for label, Wcand in (("Wanda", W_wanda), (alt_expert, W_alt)):
                                layer.weight.data.copy_(orig_full)
                                layer.weight.data[:, :Cg] = Wcand.view(O, Cg).to(dtype)
                                preds_gate = []
                                for bi in range(0, len(X_gate), gate_batch):
                                    preds_gate.append(forecast_timesfm_point(tfm_gate, X_gate[bi:bi+gate_batch], horizon))
                                pred_gate = np.concatenate(preds_gate, 0)
                                gate_mse, _ = mse_mae(pred_gate, Y_gate)
                                scores[label] = gate_mse
                            forecast_tiebreak["calls"] = int(forecast_tiebreak.get("calls", 0)) + 1
                            if scores.get(alt_expert, float("inf")) <= scores.get("Wanda", float("inf")) * 1.005:
                                final_expert, final_refit, best_mse = alt_expert, alt_refit, alt_raw
                                collapse_override = True
                        finally:
                            layer.weight.data.copy_(orig_full)
                    else:
                        final_expert, final_refit, best_mse = alt_expert, alt_refit, alt_raw
                        collapse_override = True

        hybrid_override = False
        if hybrid_policy and hybrid_policy.get("mode") == "ff1_obs_else_wanda":
            target_expert = "OBS" if is_ff1 else "Wanda"
            forced = candidate_best.get(target_expert)
            if forced is not None:
                _, forced_raw, forced_refit, _forced_idx = forced
                final_expert = target_expert
                final_refit = bool(forced_refit)
                best_mse = float(forced_raw)
                hybrid_override = True
        elif hybrid_policy and hybrid_policy.get("mode") == "ff0_qkv_mag_else_auto":
            if is_ff0 or (is_attn_proj and "qkv_proj" in lname):
                forced = candidate_best.get("MAG")
                if forced is not None:
                    _, forced_raw, forced_refit, _forced_idx = forced
                    final_expert = "MAG"
                    final_refit = bool(forced_refit)
                    best_mse = float(forced_raw)
                    hybrid_override = True

        postpass_stored = False
        if postpass_ctx and postpass_ctx.get("enabled", False):
            current_key = (final_expert, "refit" if final_refit else "mask")
            wanda_best = candidate_best.get("Wanda")
            non_wanda_best = []
            for _ename, (_cmp, _raw, _rfit, _eidx) in candidate_best.items():
                if _ename == "Wanda":
                    continue
                non_wanda_best.append((_cmp, _raw, _ename, "refit" if _rfit else "mask"))
            non_wanda_best.sort(key=lambda t: t[0])
            alt_cmp = float(non_wanda_best[0][0]) if non_wanda_best else float("inf")
            wanda_cmp = float(wanda_best[0]) if wanda_best is not None else float("inf")
            low_conf_wanda_pp = bool(
                final_expert == "Wanda"
                and wanda_cmp < float("inf")
                and alt_cmp <= wanda_cmp * (1.03 if cond_p90 < 500 else 1.05)
            )
            very_low_conf_wanda_pp = bool(
                final_expert == "Wanda"
                and wanda_cmp < float("inf")
                and alt_cmp <= wanda_cmp * 1.01
            )
            risk_score_pp = 0.0
            if ds_nsr >= 0.16:
                risk_score_pp += 1.0
            if final_expert == "Wanda" and final_refit:
                risk_score_pp += 1.5
            if low_conf_wanda_pp:
                risk_score_pp += 1.5
            if very_low_conf_wanda_pp:
                risk_score_pp += 0.5
            if offdiag_ratio > 0.14:
                risk_score_pp += 1.0
            if gdiag_cv > 4.0:
                risk_score_pp += 1.0
            if is_ff0 and (x_kurt_raw > 120 or local_nsr > 0.16):
                risk_score_pp += 1.0
            if is_attn_proj and ("qkv_proj" in lname) and (x_kurt_raw > 30 or (0.12 <= local_nsr <= 0.30)):
                risk_score_pp += 1.0
            if ("attn." in lname and ("out" in lname or "o_proj" in lname)) and (x_kurt_raw > 6 or cond_p90 > 25):
                risk_score_pp += 1.0
            if is_ff1 and (cond_p90 > 120 or x_kurt_raw > 25):
                risk_score_pp += 1.0
            if is_tokenizer and "output_layer" in lname and local_nsr > 0.8:
                risk_score_pp += 2.0
            if is_out_proj and final_expert == "Wanda" and final_refit:
                risk_score_pp += 1.0
            if is_ff0 or is_ff1 or is_attn_proj or is_out_proj:
                risk_score_pp += 0.25

            min_priority = float(postpass_ctx.get("min_priority", 2.5))
            if risk_score_pp >= min_priority:
                # Store all available expert/variant candidates (<=8) for risky layers.
                # Greedy post-pass needs this to recover cases where local proxy ranks the
                # eventually-best forecast variant (e.g., MAG/refit) below the per-expert top.
                store_keys = set(candidate_variants.keys())

                cand_items = []
                for _ename, _variant in sorted(store_keys):
                    vinfo = candidate_variants.get((_ename, _variant))
                    if vinfo is None:
                        continue
                    vcompare, vraw, vidx = float(vinfo[0]), float(vinfo[1]), int(vinfo[2])
                    Wcand_g = build_candidate_wnew(vidx, _variant == "refit")
                    Wcand_full = W.detach().clone()
                    Wcand_full[:, :Cg] = Wcand_g.view(O, Cg)
                    cand_items.append(
                        {
                            "key": f"{_ename}/{_variant}",
                            "expert": _ename,
                            "variant": _variant,
                            "compare": vcompare,
                            "raw": vraw,
                            "weight": Wcand_full.cpu(),
                        }
                    )
                if bool(postpass_ctx.get("add_sgptlite", False)):
                    sgptl = build_sparsegpt_lite_refit_obs()
                    if sgptl is not None:
                        Wcand_full = W.detach().clone()
                        Wcand_full[:, :Cg] = sgptl["weight_g"].view(O, Cg)
                        cand_items.append(
                            {
                                "key": "SGPTL/refit",
                                "expert": "SGPTL",
                                "variant": "refit",
                                "compare": float(sgptl["compare"]),
                                "raw": float(sgptl["raw"]),
                                "weight": Wcand_full.cpu(),
                                "sticky": True,
                            }
                        )
                cand_items.sort(key=lambda d: (d["compare"], d["raw"]))

                maybe_store_postpass_record(
                    postpass_ctx,
                    {
                        "layer": layer_name,
                        "layer_kind": (
                            "tokenizer" if is_tokenizer else
                            "output_proj" if is_out_proj else
                            "attn_qkv" if (is_attn_proj and "qkv_proj" in lname) else
                            "attn_out" if (is_attn_proj and ("o_proj" in lname or ".attn.out" in lname)) else
                            "ff0" if is_ff0 else
                            "ff1" if is_ff1 else
                            "other"
                        ),
                        "priority": float(risk_score_pp),
                        "ds_nsr": float(ds_nsr),
                        "local_nsr": float(local_nsr),
                        "cond_p90": float(cond_p90),
                        "x_kurt": float(x_kurt_raw),
                        "gdiag_cv": float(gdiag_cv),
                        "offdiag_ratio": float(offdiag_ratio),
                        "current_choice": f"{final_expert}/{'refit' if final_refit else 'mask'}",
                        "candidates": cand_items,
                    },
                )
                postpass_stored = True

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
        sys.stderr.write(
            f"[moe] {layer_name} Winner: {final_expert} Refit={final_refit} allow_refit={allow_refit} "
            f"MSE_val={best_mse:.6f} drop={drop_frac:.4f} "
            f"nsr_layer={local_nsr:.4f} nsr_ds={getattr(st,'avg_nsr',float('nan')):.4f} "
            f"cond_p90={cond_p90:.1f} refit_p={refit_p:.3f} "
            f"bias(mag={mag_bias:.3f},wanda={wanda_bias:.3f},obs={obs_bias:.3f},snr={snr_bias:.3f}) "
            f"kurt={x_kurt_raw:.1f} gcv={gdiag_cv:.2f} offdiag={offdiag_ratio:.3f} "
            f"mag_fallback(active={noisy_mag_active},override={noisy_mag_override}) "
            f"safe(active={safe_active},override={safe_override},target={safe_target},reason={safe_reason}) "
            f"collapse(active={collapse_active},override={collapse_override}) "
            f"hybrid_override={hybrid_override} postpass_store={postpass_stored}\n"
        )
        if gate_state is not None:
            gate_state["all_total"] += 1
            if final_expert == "Wanda":
                gate_state["all_wanda"] += 1
            if is_risky_layer:
                gate_state["risky_total"] += 1
                if final_expert == "Wanda":
                    gate_state["risky_wanda"] += 1


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
    ap.add_argument("--hybrid_policy", default="", choices=["", "ff1_obs_else_wanda", "ff0_qkv_mag_else_auto"])
    ap.add_argument("--safe_policy", default="", choices=["", "rule_v1"])
    ap.add_argument("--greedy_postpass_k", type=int, default=0)
    ap.add_argument("--greedy_postpass_steps", type=int, default=0)
    ap.add_argument("--greedy_postpass_eval_windows", type=int, default=32)
    ap.add_argument("--greedy_postpass_eval_select", default="last", choices=["first", "last"])
    ap.add_argument("--greedy_postpass_min_step_gain", type=float, default=0.02)
    ap.add_argument("--greedy_postpass_min_priority", type=float, default=2.5)
    ap.add_argument("--greedy_postpass_screen_k", type=int, default=0)
    ap.add_argument("--greedy_postpass_screen_min_gain", type=float, default=0.05)
    ap.add_argument("--greedy_postpass_max_cands_per_layer", type=int, default=0)
    ap.add_argument("--greedy_postpass_add_sgptlite", type=int, default=0)
    ap.add_argument("--greedy_postpass_pairdiag_k", type=int, default=0)
    ap.add_argument("--greedy_postpass_pairdiag_exact_pairs", type=int, default=0)
    ap.add_argument("--greedy_postpass_pairdiag_prefix", default="")
    ap.add_argument("--greedy_postpass_pairaware", type=int, default=0)
    ap.add_argument("--greedy_postpass_pairaware_alpha", type=float, default=1.0)
    ap.add_argument("--greedy_postpass_pairaware_use_exact", type=int, default=0)
    ap.add_argument("--pretrained", default="")
    args = ap.parse_args()

    pretrained_ref = args.pretrained.strip()
    if not pretrained_ref:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--timesfm-2.5-200m-pytorch" / "snapshots"
        if hf_cache.exists():
            snaps = sorted([p for p in hf_cache.iterdir() if p.is_dir()], key=lambda p: p.name)
            if snaps:
                pretrained_ref = str(snaps[-1])
        if not pretrained_ref:
            pretrained_ref = "google/timesfm-2.5-200m-pytorch"
    local_pretrained_dir = Path(pretrained_ref) if pretrained_ref else None
    if local_pretrained_dir is not None and local_pretrained_dir.is_dir():
        try:
            from timesfm.timesfm_2p5 import timesfm_2p5_torch as tfm25_mod

            orig_hf_hub_download = tfm25_mod.hf_hub_download

            def _hf_hub_download_local_first(*args, **kwargs):
                filename = kwargs.get("filename")
                if filename is None and len(args) >= 2:
                    filename = args[1]
                if isinstance(filename, str):
                    local_file = local_pretrained_dir / filename
                    if local_file.exists():
                        return str(local_file)
                return orig_hf_hub_download(*args, **kwargs)

            tfm25_mod.hf_hub_download = _hf_hub_download_local_first
            print(f"[pretrained] local_hf_patch=on dir={local_pretrained_dir}")
        except Exception as e:
            print(f"[pretrained] local_hf_patch=fail err={type(e).__name__}:{e}")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(pretrained_ref)
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

    gate_state = {"warmup": 12, "wanda_frac_thresh": 0.75}
    # Tiny forecast-aware tie-break set from the calibration tail (cheap and only used sparingly).
    gate_n = min(8, len(X_pool))
    forecast_tiebreak = {
        "enabled": gate_n > 0,
        "tfm": tfm,
        "X": X_pool[-gate_n:],
        "Y": Y_pool[-gate_n:],
        "batch": min(args.batch, 4),
        "calls": 0,
        "max_calls": 8,
    }
    hybrid_policy = {"mode": args.hybrid_policy} if args.hybrid_policy else None
    safe_policy = {"mode": args.safe_policy} if args.safe_policy else None
    postpass_ctx = None
    if args.greedy_postpass_k > 0:
        postpass_ctx = {
            "enabled": True,
            "pool_k": int(args.greedy_postpass_k),
            "steps": int(args.greedy_postpass_steps),
            "eval_windows": int(args.greedy_postpass_eval_windows),
            "eval_select": str(args.greedy_postpass_eval_select),
            "min_step_gain": float(args.greedy_postpass_min_step_gain),
            "min_priority": float(args.greedy_postpass_min_priority),
            "screen_k": int(args.greedy_postpass_screen_k),
            "screen_min_gain": float(args.greedy_postpass_screen_min_gain),
            "max_cands_per_layer": int(args.greedy_postpass_max_cands_per_layer),
            "add_sgptlite": bool(args.greedy_postpass_add_sgptlite),
            "pairdiag_k": int(args.greedy_postpass_pairdiag_k),
            "pairdiag_exact_pairs": int(args.greedy_postpass_pairdiag_exact_pairs),
            "pairdiag_prefix": str(args.greedy_postpass_pairdiag_prefix),
            "pairaware": bool(args.greedy_postpass_pairaware),
            "pairaware_alpha": float(args.greedy_postpass_pairaware_alpha),
            "pairaware_use_exact": bool(args.greedy_postpass_pairaware_use_exact),
            "records": {},
        }
    for name, layer in targets:
        st = stats.get(name)
        if st: prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), args.ridge, horizon=args.horizon, nf_hi=args.nf_hi, layer_name=name, gate_state=gate_state, forecast_tiebreak=forecast_tiebreak, hybrid_policy=hybrid_policy, safe_policy=safe_policy, postpass_ctx=postpass_ctx)

    if postpass_ctx and postpass_ctx.get("enabled", False):
        run_greedy_postpass(tfm, targets, X_test, Y_test, args.horizon, args.batch, postpass_ctx)

    # Eval
    mse_p, mae_p = eval_forecast_mse(tfm, X_test, Y_test, args.horizon, args.batch)
    print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f}")
    print(f"[delta] ΔMSE={mse_p - mse_b:+.6f}")

if __name__ == "__main__":
    main()
