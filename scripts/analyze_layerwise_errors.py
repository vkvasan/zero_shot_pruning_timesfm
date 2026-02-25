#!/usr/bin/env python3
"""
Forecast-aware layerwise expert attribution for Unified pruning.

For a target config, this script:
1) builds the current Unified-pruned model,
2) generates per-layer expert candidates (MAG/Wanda/SNR/OBS, mask/refit),
3) swaps one layer at a time on top of the Unified model,
4) measures forecast MSE deltas on an evaluation set.

This isolates which expert is best *for each layer* under the current model
and highlights where Unified's local proxy may be mis-ranking experts.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import timesfm

from prune_unified import (
    PAIRS,
    PAIR_MASKS,
    collect_stats,
    find_torch_module,
    forecast_timesfm_point,
    load_series,
    make_windows,
    mse_mae,
    prune_linear_snr_2of4,
    select_linears,
)


def _layer_kind(name: str) -> str:
    lname = name.lower()
    if "tokenizer." in lname:
        return "tokenizer"
    if "output_projection_point" in lname:
        return "output_proj"
    if ".attn." in lname and "qkv_proj" in lname:
        return "attn_qkv"
    if ".attn." in lname and ("out" in lname or "o_proj" in lname):
        return "attn_out"
    if ".ff0" in lname:
        return "ff0"
    if ".ff1" in lname:
        return "ff1"
    return "other"


def _eval_mse(tfm, X: np.ndarray, Y: np.ndarray, horizon: int, batch: int) -> Tuple[float, float]:
    preds = []
    for i in range(0, len(X), batch):
        preds.append(forecast_timesfm_point(tfm, X[i : i + batch], horizon))
    return mse_mae(np.concatenate(preds, 0), Y)


def _compute_avg_nsr(stats: Dict[str, object]) -> float:
    nsrs = []
    for st in stats.values():
        te = float(st.trend_energy.sum()) if st.trend_energy is not None else 0.0
        se = float(st.season_energy.sum()) if st.season_energy is not None else 0.0
        ne = float(st.noise_energy.sum()) if st.noise_energy is not None else 0.0
        nsrs.append(ne / (te + se + 1e-9))
    return float(np.mean(nsrs)) if nsrs else float("nan")


def _expert_candidates_for_layer(
    W_full_orig: torch.Tensor,
    st,
    ridge: float,
    eps: float,
    score_mode: str = "unified",
) -> dict:
    if score_mode != "unified":
        raise ValueError("Only score_mode=unified is supported by this analysis script.")

    W = W_full_orig
    O, C = W.shape
    Ggroups = C // 4
    Cg = Ggroups * 4
    if Cg == 0:
        return {"skip": True}

    device = W.device
    dtype = W.dtype
    Wg = W[:, :Cg].view(O, Ggroups, 4)
    Gs = (st.Gsig / max(st.Csig, 1e-6)).to(device=device, dtype=dtype)
    Ga = (st.Gact / max(st.Cact, 1e-6)).to(device=device, dtype=dtype)
    Gn = (st.Gnoi / max(st.Cnoi, 1e-6)).to(device=device, dtype=dtype) if st.Gnoi is not None and st.Cnoi > 0 else None

    damp_s = 0.01 * torch.mean(torch.diagonal(Gs, dim1=1, dim2=2))
    Hinv_s = torch.inverse(Gs + damp_s * torch.eye(4, device=device, dtype=dtype).unsqueeze(0))
    diag_s = torch.diagonal(Hinv_s, dim1=1, dim2=2)

    damp_a = 0.01 * torch.mean(torch.diagonal(Ga, dim1=1, dim2=2))
    Hinv_a = torch.inverse(Ga + damp_a * torch.eye(4, device=device, dtype=dtype).unsqueeze(0))
    diag_a = torch.diagonal(Hinv_a, dim1=1, dim2=2)
    act_diag = torch.diagonal(Ga, dim1=1, dim2=2).clamp_min(1e-8)

    log_ridge = math.log10(max(ridge, 1e-10))
    t_ridge = max(0.0, min(1.0, (log_ridge - (-3.0)) / 1.0))
    w_obs, w_ratio, w_mag = 0.25 + 0.6 * t_ridge, 0.5 - 0.4 * t_ridge, 0.25 - 0.2 * t_ridge

    def znorm(t):
        return (t - t.mean()) / (t.std() + 1e-9)

    masks_t = PAIR_MASKS.to(device=device, dtype=dtype)
    wanda_imp = (Wg ** 2) * act_diag.unsqueeze(0)
    obs_imp_a = (Wg ** 2) / (diag_a.unsqueeze(0) + 1e-10)

    scores_mag = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
    scores_wanda = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
    scores_obs = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
    snr_ratio = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
    snr_mag = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
    snr_obsig = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)

    for k in range(6):
        mk = masks_t[k].view(1, 1, 4)
        Wk, Wd = Wg * mk, Wg * (1.0 - mk)
        scores_mag[:, :, k] = Wk.abs().sum(dim=2)
        scores_wanda[:, :, k] = (wanda_imp * mk).sum(dim=2)
        scores_obs[:, :, k] = (obs_imp_a * mk).sum(dim=2)

        Tk_s = torch.einsum("ogc,gcd->ogd", Wk, Gs)
        Td_s = torch.einsum("ogc,gcd->ogd", Wd, Gs)
        s_ratio = (Tk_s * Wk).sum(dim=2) / ((Td_s * Wd).sum(dim=2) + eps)

        if Gn is not None:
            Tk_n = torch.einsum("ogc,gcd->ogd", Wk, Gn)
            Td_n = torch.einsum("ogc,gcd->ogd", Wd, Gn)
            n_ratio = (Tk_n * Wk).sum(dim=2) / ((Td_n * Wd).sum(dim=2) + eps)
            snr_ratio[:, :, k] = s_ratio / (n_ratio + eps)
        else:
            snr_ratio[:, :, k] = s_ratio

        snr_mag[:, :, k] = Wk.abs().sum(dim=2)
        snr_obsig[:, :, k] = (((Wk ** 2) / (diag_s.unsqueeze(0) + 1e-10)) * mk).sum(dim=2)

    scores_snr = (
        w_ratio * znorm(snr_ratio.reshape(-1, 6))
        + w_mag * znorm(snr_mag.reshape(-1, 6))
        + w_obs * znorm(snr_obsig.reshape(-1, 6))
    ).reshape(O, Ggroups, 6)
    expert_bestks = [
        torch.argmax(scores_mag, 2),
        torch.argmax(scores_wanda, 2),
        torch.argmax(scores_snr, 2),
        torch.argmax(scores_obs, 2),
    ]
    expert_names = ["MAG", "Wanda", "SNR", "OBS"]

    X_val_d = st.X_val.to(device=device, dtype=dtype) if st.X_val is not None else torch.randn((1, Cg), device=device, dtype=dtype)
    Y_val_dense = (X_val_d @ W[:, :Cg].T).detach()
    H_reg = Gs + ridge * torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    B = torch.einsum("ogc,gcd->ogd", Wg, H_reg)
    invs = torch.stack([torch.inverse(Gs[:, PAIRS[k]][:, :, PAIRS[k]] + ridge * torch.eye(2, device=device, dtype=dtype)) for k in range(6)])
    g_idx = torch.arange(Ggroups, device=device).view(1, Ggroups).expand(O, Ggroups)

    def _build_mask(bestk):
        mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool).scatter_(2, PAIRS.to(device)[bestk], True)
        return torch.where(mask, Wg, torch.zeros_like(Wg))

    def _build_refit(bestk):
        Wr = torch.zeros_like(Wg)
        for k in range(6):
            sel = bestk == k
            if not torch.any(sel):
                continue
            bsel = torch.stack([B[:, :, PAIRS[k, 0]][sel], B[:, :, PAIRS[k, 1]][sel]], 1)
            u = torch.bmm(invs[k, g_idx[sel]], bsel.unsqueeze(2)).squeeze(2)
            Wr[:, :, PAIRS[k, 0]][sel], Wr[:, :, PAIRS[k, 1]][sel] = u[:, 0], u[:, 1]
        return Wr

    candidates = {}
    for name, bestk in zip(expert_names, expert_bestks):
        Wm_g = _build_mask(bestk)
        mse_mask = torch.mean(((X_val_d @ Wm_g.view(O, Cg).T) - Y_val_dense) ** 2).item()

        Wr_g = _build_refit(bestk)
        mse_refit = torch.mean(((X_val_d @ Wr_g.view(O, Cg).T) - Y_val_dense) ** 2).item()

        Wm_full = W.clone()
        Wm_full[:, :Cg] = Wm_g.view(O, Cg)
        Wr_full = W.clone()
        Wr_full[:, :Cg] = Wr_g.view(O, Cg)

        candidates[name] = {
            "mask": {"weight": Wm_full, "local_mse": float(mse_mask)},
            "refit": {"weight": Wr_full, "local_mse": float(mse_refit)},
        }

    eye4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    ga_cond = torch.linalg.cond(Ga + 1e-6 * eye4)
    cond_p90 = float(torch.quantile(ga_cond.float(), 0.90).item())
    te = float(st.trend_energy.sum()) if st.trend_energy is not None else 0.0
    se = float(st.season_energy.sum()) if st.season_energy is not None else 0.0
    ne = float(st.noise_energy.sum()) if st.noise_energy is not None else 0.0
    local_nsr = ne / (te + se + 1e-9)

    return {
        "skip": False,
        "candidates": candidates,
        "features": {
            "local_nsr": float(local_nsr),
            "cond_p90": float(cond_p90),
        },
    }


def _identify_current_variant(current_w: torch.Tensor, candidates: dict) -> Tuple[str, str, float]:
    best = ("unknown", "unknown", float("inf"))
    for expert, variants in candidates.items():
        for variant, info in variants.items():
            dw = (current_w - info["weight"]).float()
            denom = float(current_w.float().norm().item()) + 1e-12
            rel = float(dw.norm().item() / denom)
            if rel < best[2]:
                best = (expert, variant, rel)
    return best


def _variant_keys_from_row(row: dict, include_refit: bool) -> List[Tuple[str, str, float]]:
    keys = []
    for expert in ("MAG", "Wanda", "SNR", "OBS"):
        for variant in (["mask", "refit"] if include_refit else ["mask"]):
            k = f"mse_{expert}_{variant}"
            if k in row and row[k] is not None:
                keys.append((expert, variant, float(row[k])))
    keys.sort(key=lambda t: t[2])
    return keys


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
    ap.add_argument("--eval_windows", type=int, default=0, help="0 = full test set")
    ap.add_argument("--eval_select", choices=["first", "last"], default="last")
    ap.add_argument("--max_layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--greedy_topk", type=int, default=0, help="Run greedy multi-layer search on top-K layers by one-layer gain (0 disables)")
    ap.add_argument("--greedy_steps", type=int, default=0, help="Max greedy override steps (0 => use greedy_topk)")
    ap.add_argument("--greedy_candidates_per_layer", type=int, default=2, help="Evaluate top-N candidate variants per selected layer in greedy search")
    ap.add_argument("--greedy_preselect_min_gain", type=float, default=0.05, help="Minimum one-layer gain to enter greedy candidate pool")
    ap.add_argument("--greedy_min_step_gain", type=float, default=0.02, help="Minimum eval-MSE improvement required per greedy step")
    ap.add_argument("--greedy_eval_full", type=int, default=1, help="After greedy on eval subset, evaluate on full test set")
    ap.add_argument("--out_dir", default="results/layerwise")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    torch_mod = find_torch_module(tfm)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=args.horizon))

    series = load_series(args.csv, args.col)
    X_train, Y_train = make_windows(series, 0, args.train_end, args.context, args.horizon, 1)
    X_test, Y_test = make_windows(series, args.train_end, len(series), args.context, args.horizon, args.stride_test)
    if args.eval_windows and args.eval_windows < len(X_test):
        if args.eval_select == "last":
            X_eval, Y_eval = X_test[-args.eval_windows :], Y_test[-args.eval_windows :]
        else:
            X_eval, Y_eval = X_test[: args.eval_windows], Y_test[: args.eval_windows]
    else:
        X_eval, Y_eval = X_test, Y_test

    dense_mse, dense_mae = _eval_mse(tfm, X_eval, Y_eval, args.horizon, args.batch)
    print(f"[baseline-eval] windows={len(X_eval)} MSE={dense_mse:.6f} MAE={dense_mae:.6f}")

    X_pool = X_train[-1024:] if args.calib_select == "last" else X_train[:1024]
    Y_pool = Y_train[-1024:] if args.calib_select == "last" else Y_train[:1024]

    preds_pool = []
    for i in range(0, len(X_pool), args.calib_batch):
        preds_pool.append(forecast_timesfm_point(tfm, X_pool[i : i + args.calib_batch], args.horizon))
    errs = np.mean((np.concatenate(preds_pool, 0) - Y_pool) ** 2, axis=1)
    weights_sig = (errs / (errs.mean() + 1e-7)) ** args.error_power
    weights_noi = (1.0 / (errs + 1e-7)) / ((1.0 / (errs + 1e-7)).mean() + 1e-7)
    weights_noi = weights_noi ** args.error_power

    targets_all = select_linears(torch_mod, False, ".*")
    targets = targets_all[: args.max_layers] if args.max_layers > 0 else targets_all
    print(f"[targets] total={len(targets_all)} using={len(targets)}")
    stats = collect_stats(tfm, targets, X_pool, weights_sig, weights_noi, args.horizon, args.calib_batch, args.max_calls_per_layer)
    avg_nsr = _compute_avg_nsr(stats)
    for st in stats.values():
        st.avg_nsr = avg_nsr
    print(f"[diag] avg_nsr={avg_nsr:.4f}")

    # Save dense weights for candidate generation.
    orig_weights: Dict[str, torch.Tensor] = {name: layer.weight.data.detach().clone() for name, layer in targets}

    # Build current Unified-pruned model (suppress verbose stderr logs).
    gate_state = {"warmup": 12, "wanda_frac_thresh": 0.75}
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
    with contextlib.redirect_stderr(io.StringIO()):
        for name, layer in targets:
            st = stats.get(name)
            if st is not None:
                prune_linear_snr_2of4(
                    layer,
                    st,
                    args.score_mode,
                    args.eps,
                    bool(args.refit),
                    args.ridge,
                    horizon=args.horizon,
                    nf_hi=args.nf_hi,
                    layer_name=name,
                    gate_state=gate_state,
                    forecast_tiebreak=forecast_tiebreak,
                    hybrid_policy=None,
                )

    unified_mse, unified_mae = _eval_mse(tfm, X_eval, Y_eval, args.horizon, args.batch)
    print(f"[unified-eval] MSE={unified_mse:.6f} MAE={unified_mae:.6f} delta={unified_mse - dense_mse:+.6f}")

    rows: List[dict] = []
    rows_by_layer: Dict[str, dict] = {}
    kind_counts = {}
    best_expert_counts = {}

    for idx, (name, layer) in enumerate(targets, start=1):
        st = stats.get(name)
        if st is None:
            continue
        current_w = layer.weight.data.detach().clone()
        kind = _layer_kind(name)

        layer_pack = _expert_candidates_for_layer(orig_weights[name], st, args.ridge, args.eps, score_mode=args.score_mode)
        if layer_pack.get("skip", False):
            continue
        candidates = layer_pack["candidates"]
        feats = layer_pack["features"]

        cur_expert, cur_variant, cur_match_err = _identify_current_variant(current_w, candidates)

        variant_mses = {}
        # One-layer swap on top of current unified model.
        for expert in ("MAG", "Wanda", "SNR", "OBS"):
            for variant in (["mask", "refit"] if args.refit else ["mask"]):
                Wcand = candidates[expert][variant]["weight"]
                layer.weight.data.copy_(Wcand.to(dtype=layer.weight.data.dtype, device=layer.weight.data.device))
                mse_swap, _ = _eval_mse(tfm, X_eval, Y_eval, args.horizon, args.batch)
                variant_mses[(expert, variant)] = float(mse_swap)

        layer.weight.data.copy_(current_w)

        # Best variant per expert and best expert overall.
        expert_best = {}
        for expert in ("MAG", "Wanda", "SNR", "OBS"):
            opts = [(variant, variant_mses[(expert, variant)]) for variant in (["mask", "refit"] if args.refit else ["mask"])]
            best_variant, best_mse = min(opts, key=lambda t: t[1])
            expert_best[expert] = (best_variant, float(best_mse))
        best_expert, (best_variant, best_swap_mse) = min(expert_best.items(), key=lambda kv: kv[1][1])
        cur_swap_mse = variant_mses.get((cur_expert, cur_variant), unified_mse)

        row = {
            "layer": name,
            "layer_kind": kind,
            "idx": idx,
            "local_nsr": feats["local_nsr"],
            "cond_p90": feats["cond_p90"],
            "current_expert": cur_expert,
            "current_variant": cur_variant,
            "current_match_relerr": cur_match_err,
            "unified_eval_mse": unified_mse,
            "current_swap_eval_mse": cur_swap_mse,
            "best_expert_eval": best_expert,
            "best_variant_eval": best_variant,
            "best_swap_eval_mse": best_swap_mse,
            "gain_vs_current_choice": cur_swap_mse - best_swap_mse,
            "gain_vs_unified": unified_mse - best_swap_mse,
        }
        for expert in ("MAG", "Wanda", "SNR", "OBS"):
            for variant in (["mask", "refit"] if args.refit else ["mask"]):
                row[f"mse_{expert}_{variant}"] = variant_mses[(expert, variant)]
                row[f"local_{expert}_{variant}"] = candidates[expert][variant]["local_mse"]
            row[f"mse_{expert}_best"] = expert_best[expert][1]
            row[f"variant_{expert}_best"] = expert_best[expert][0]

        rows.append(row)
        rows_by_layer[name] = row
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        best_expert_counts[best_expert] = best_expert_counts.get(best_expert, 0) + 1

        print(
            f"[{idx:03d}/{len(targets)}] {name} kind={kind} "
            f"cur={cur_expert}/{cur_variant} best={best_expert}/{best_variant} "
            f"gain={row['gain_vs_current_choice']:+.6f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{Path(args.csv).stem}_h{args.horizon}_c{args.context}"
    csv_path = out_dir / f"{tag}_layerwise_error_attribution.csv"
    json_path = out_dir / f"{tag}_layerwise_error_attribution_summary.json"

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ordered_gains = sorted(rows, key=lambda r: r["gain_vs_current_choice"], reverse=True)
    summary = {
        "config": {
            "csv": args.csv,
            "col": args.col,
            "train_end": args.train_end,
            "context": args.context,
            "horizon": args.horizon,
            "stride_test": args.stride_test,
            "eval_windows": int(len(X_eval)),
        },
        "metrics": {
            "dense_eval_mse": dense_mse,
            "dense_eval_mae": dense_mae,
            "unified_eval_mse": unified_mse,
            "unified_eval_mae": unified_mae,
            "unified_delta_mse": unified_mse - dense_mse,
            "avg_nsr": avg_nsr,
        },
        "counts": {
            "layers": len(rows),
            "layer_kind": kind_counts,
            "best_expert_eval": best_expert_counts,
        },
        "top_gains": [
            {
                "layer": r["layer"],
                "layer_kind": r["layer_kind"],
                "current": f"{r['current_expert']}/{r['current_variant']}",
                "best": f"{r['best_expert_eval']}/{r['best_variant_eval']}",
                "gain_vs_current_choice": r["gain_vs_current_choice"],
                "local_nsr": r["local_nsr"],
                "cond_p90": r["cond_p90"],
            }
            for r in ordered_gains[:20]
        ],
    }

    if args.greedy_topk > 0 and rows:
        layer_map = {name: layer for name, layer in targets}
        ordered = sorted(
            [r for r in rows if r["gain_vs_current_choice"] >= args.greedy_preselect_min_gain],
            key=lambda r: r["gain_vs_current_choice"],
            reverse=True,
        )
        selected_rows = ordered[: args.greedy_topk]
        selected_names = [r["layer"] for r in selected_rows]
        greedy_steps_max = args.greedy_steps if args.greedy_steps > 0 else len(selected_names)
        chosen_overrides = {}
        greedy_log = []
        current_eval_mse = unified_mse
        remaining = set(selected_names)

        print(
            f"[greedy] start eval_mse={current_eval_mse:.6f} "
            f"pool={len(selected_names)} steps_max={greedy_steps_max}"
        )

        # Keep a snapshot of unified-pruned weights to support restore/re-eval.
        unified_weights = {name: layer_map[name].weight.data.detach().clone() for name in selected_names}

        for step in range(1, greedy_steps_max + 1):
            best_move = None
            for lname in list(remaining):
                row = rows_by_layer[lname]
                st = stats.get(lname)
                layer = layer_map[lname]
                if st is None:
                    continue
                variant_ranked = _variant_keys_from_row(row, bool(args.refit))
                current_key = (row["current_expert"], row["current_variant"])
                cand_keys = []
                for expert, variant, _mse in variant_ranked:
                    key = (expert, variant)
                    if key == current_key:
                        continue
                    cand_keys.append(key)
                    if len(cand_keys) >= args.greedy_candidates_per_layer:
                        break
                if not cand_keys:
                    continue

                cand_pack = _expert_candidates_for_layer(orig_weights[lname], st, args.ridge, args.eps, score_mode=args.score_mode)
                if cand_pack.get("skip", False):
                    continue
                candidates = cand_pack["candidates"]
                orig_layer_w = layer.weight.data.detach().clone()
                try:
                    for expert, variant in cand_keys:
                        Wcand = candidates[expert][variant]["weight"]
                        layer.weight.data.copy_(Wcand.to(dtype=layer.weight.data.dtype, device=layer.weight.data.device))
                        mse_try, _ = _eval_mse(tfm, X_eval, Y_eval, args.horizon, args.batch)
                        gain = current_eval_mse - mse_try
                        if (best_move is None) or (gain > best_move["gain"]):
                            best_move = {
                                "layer": lname,
                                "expert": expert,
                                "variant": variant,
                                "mse": float(mse_try),
                                "gain": float(gain),
                                "one_layer_gain_hint": float(row["gain_vs_current_choice"]),
                                "layer_kind": row["layer_kind"],
                            }
                finally:
                    layer.weight.data.copy_(orig_layer_w)

            if best_move is None or best_move["gain"] < args.greedy_min_step_gain:
                print(
                    f"[greedy] stop step={step} "
                    f"best_gain={(best_move['gain'] if best_move else float('nan')):.6f}"
                )
                break

            lname = best_move["layer"]
            layer = layer_map[lname]
            st = stats[lname]
            cand_pack = _expert_candidates_for_layer(orig_weights[lname], st, args.ridge, args.eps, score_mode=args.score_mode)
            Wapply = cand_pack["candidates"][best_move["expert"]][best_move["variant"]]["weight"]
            layer.weight.data.copy_(Wapply.to(dtype=layer.weight.data.dtype, device=layer.weight.data.device))
            current_eval_mse = float(best_move["mse"])
            remaining.remove(lname)
            chosen_overrides[lname] = (best_move["expert"], best_move["variant"])
            greedy_log.append(best_move)
            print(
                f"[greedy] step={step} apply {lname} -> {best_move['expert']}/{best_move['variant']} "
                f"gain={best_move['gain']:+.6f} eval_mse={current_eval_mse:.6f}"
            )

        summary["greedy_search"] = {
            "enabled": True,
            "pool_size": len(selected_names),
            "steps_max": greedy_steps_max,
            "steps_applied": len(greedy_log),
            "eval_mse_before": unified_mse,
            "eval_mse_after": current_eval_mse,
            "eval_gain_total": unified_mse - current_eval_mse,
            "moves": greedy_log,
        }

        if args.greedy_eval_full:
            full_mse_before = None
            full_mae_before = None
            if len(X_eval) != len(X_test):
                # Reconstruct unified baseline on full test before comparing.
                # Model is still in the unified-pruned state here unless greedy applied steps.
                # If greedy applied, restore selected layers temporarily.
                for lname in selected_names:
                    layer_map[lname].weight.data.copy_(unified_weights[lname])
                full_mse_before, full_mae_before = _eval_mse(tfm, X_test, Y_test, args.horizon, args.batch)
                for move in greedy_log:
                    lname = move["layer"]
                    st = stats[lname]
                    cand_pack = _expert_candidates_for_layer(orig_weights[lname], st, args.ridge, args.eps, score_mode=args.score_mode)
                    Wapply = cand_pack["candidates"][move["expert"]][move["variant"]]["weight"]
                    layer_map[lname].weight.data.copy_(Wapply.to(dtype=layer_map[lname].weight.data.dtype, device=layer_map[lname].weight.data.device))
            else:
                full_mse_before, full_mae_before = unified_mse, unified_mae

            full_mse_after, full_mae_after = _eval_mse(tfm, X_test, Y_test, args.horizon, args.batch)
            print(
                f"[greedy-full] MSE={full_mse_after:.6f} MAE={full_mae_after:.6f} "
                f"delta_vs_unified={full_mse_after - full_mse_before:+.6f}"
            )
            summary["greedy_search"]["full_test_after"] = {
                "mse": full_mse_after,
                "mae": full_mae_after,
            }
            summary["greedy_search"]["full_test_before"] = {
                "mse": full_mse_before,
                "mae": full_mae_before,
            }
        else:
            # Restore selected layers to original unified state for cleanliness.
            for lname in selected_names:
                layer_map[lname].weight.data.copy_(unified_weights[lname])
    else:
        summary["greedy_search"] = {"enabled": False}
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[out] {csv_path}")
    print(f"[out] {json_path}")


if __name__ == "__main__":
    main()
