#!/usr/bin/env python3
"""
Layer-wise activation/weight distribution diagnostics for TimesFM pruning.

This script reuses the calibration/stat-collection path from `prune_unified.py`
but does not prune. It emits per-layer distribution summaries that are useful
for designing better MoE gating rules.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import timesfm

from prune_unified import (
    PAIR_MASKS,
    collect_stats,
    find_torch_module,
    forecast_timesfm_point,
    load_series,
    make_windows,
    mse_mae,
    select_linears,
)


def _safe_quantiles(x: np.ndarray, qs):
    if x.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(x, q)) for q in qs]


def _winner_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts.astype(np.float64) / float(total)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _pair_score_summary(Wg: torch.Tensor, Gs: torch.Tensor, Ga: torch.Tensor, Gn: torch.Tensor | None, eps: float, ridge: float):
    device = Wg.device
    dtype = Wg.dtype
    O, G, _ = Wg.shape
    masks = PAIR_MASKS.to(device=device, dtype=dtype)

    damp_a = 0.01 * torch.mean(torch.diagonal(Ga, dim1=1, dim2=2))
    Hinv_a = torch.inverse(Ga + damp_a * torch.eye(4, device=device, dtype=dtype).unsqueeze(0))
    diag_a = torch.diagonal(Hinv_a, dim1=1, dim2=2)
    act_diag = torch.diagonal(Ga, dim1=1, dim2=2).clamp_min(1e-8)

    damp_s = 0.01 * torch.mean(torch.diagonal(Gs, dim1=1, dim2=2))
    Hinv_s = torch.inverse(Gs + damp_s * torch.eye(4, device=device, dtype=dtype).unsqueeze(0))
    diag_s = torch.diagonal(Hinv_s, dim1=1, dim2=2)

    wanda_imp = (Wg ** 2) * act_diag.unsqueeze(0)
    obs_imp_a = (Wg ** 2) / (diag_a.unsqueeze(0) + 1e-10)

    scores = {
        "mag": torch.zeros((O, G, 6), device=device, dtype=dtype),
        "wanda": torch.zeros((O, G, 6), device=device, dtype=dtype),
        "obs": torch.zeros((O, G, 6), device=device, dtype=dtype),
        "snr": torch.zeros((O, G, 6), device=device, dtype=dtype),
    }
    snr_ratio = torch.zeros((O, G, 6), device=device, dtype=dtype)
    snr_mag = torch.zeros((O, G, 6), device=device, dtype=dtype)
    snr_obsig = torch.zeros((O, G, 6), device=device, dtype=dtype)

    def znorm(t):
        return (t - t.mean()) / (t.std() + 1e-9)

    log_ridge = math.log10(max(ridge, 1e-10))
    t_ridge = max(0.0, min(1.0, (log_ridge - (-3.0)) / 1.0))
    w_obs, w_ratio, w_mag = 0.25 + 0.6 * t_ridge, 0.5 - 0.4 * t_ridge, 0.25 - 0.2 * t_ridge

    for k in range(6):
        mk = masks[k].view(1, 1, 4)
        Wk, Wd = Wg * mk, Wg * (1.0 - mk)
        scores["mag"][:, :, k] = Wk.abs().sum(dim=2)
        scores["wanda"][:, :, k] = (wanda_imp * mk).sum(dim=2)
        scores["obs"][:, :, k] = (obs_imp_a * mk).sum(dim=2)

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

    scores["snr"] = (
        w_ratio * znorm(snr_ratio.reshape(-1, 6))
        + w_mag * znorm(snr_mag.reshape(-1, 6))
        + w_obs * znorm(snr_obsig.reshape(-1, 6))
    ).reshape(O, G, 6)

    out = {}
    winners = {}
    for name, s in scores.items():
        top2 = torch.topk(s, 2, dim=2).values
        margin = ((top2[:, :, 0] - top2[:, :, 1]) / (top2[:, :, 0].abs() + eps)).detach().cpu().numpy().reshape(-1)
        win = torch.argmax(s, dim=2).detach().cpu().numpy().reshape(-1)
        winners[name] = win
        counts = np.bincount(win, minlength=6)
        out[f"{name}_winner_entropy"] = _winner_entropy(counts)
        out[f"{name}_margin_mean"] = float(np.mean(margin))
        out[f"{name}_margin_p10"], out[f"{name}_margin_p50"], out[f"{name}_margin_p90"] = _safe_quantiles(margin, [0.1, 0.5, 0.9])
        for i in range(6):
            out[f"{name}_pair{i}_freq"] = float(counts[i] / max(1, counts.sum()))

    names = ["mag", "wanda", "snr", "obs"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[f"{a}_{b}_agree_rate"] = float(np.mean(winners[a] == winners[b]))

    return out


def _layer_distribution_row(name: str, layer, st, ridge: float, eps: float):
    W = layer.weight.detach().float().cpu()
    O, C = W.shape
    G = C // 4
    Cg = G * 4

    row = {
        "layer": name,
        "out_features": int(O),
        "in_features": int(C),
        "groups_2of4": int(G),
    }

    w_abs = W.abs().numpy().reshape(-1)
    row["w_abs_mean"] = float(w_abs.mean())
    row["w_abs_std"] = float(w_abs.std())
    row["w_abs_p50"], row["w_abs_p90"], row["w_abs_p99"] = _safe_quantiles(w_abs, [0.5, 0.9, 0.99])
    row["w_abs_max"] = float(w_abs.max()) if w_abs.size else float("nan")

    if Cg == 0 or st is None:
        row["has_stats"] = 0
        return row

    row["has_stats"] = 1
    Wg = W[:, :Cg].view(O, G, 4)

    group_abs = Wg.abs()
    top = torch.topk(group_abs, 3, dim=2).values
    top1 = top[:, :, 0]
    top2 = top[:, :, 1]
    top3 = top[:, :, 2]
    row["w_top1_top2_margin_mean"] = float(((top1 - top2) / (top1 + eps)).mean().item())
    row["w_top2_top3_margin_mean"] = float(((top2 - top3) / (top2 + eps)).mean().item())
    row["w_group_l2_mean"] = float(torch.linalg.norm(Wg, dim=2).mean().item())
    row["w_group_l2_cv"] = float((torch.linalg.norm(Wg, dim=2).std() / (torch.linalg.norm(Wg, dim=2).mean() + eps)).item())

    # Activation moments from collected validation activations.
    if st.X_val is not None:
        xv = st.X_val.float().numpy().reshape(-1)
        xv_abs = np.abs(xv)
        row["x_abs_mean"] = float(xv_abs.mean())
        row["x_abs_std"] = float(xv_abs.std())
        row["x_abs_p50"], row["x_abs_p90"], row["x_abs_p99"] = _safe_quantiles(xv_abs, [0.5, 0.9, 0.99])
        x2 = float(np.mean(xv * xv))
        x4 = float(np.mean((xv * xv) ** 2))
        row["x_kurtosis_raw"] = float(x4 / (x2 * x2 + eps))
    else:
        row["x_abs_mean"] = row["x_abs_std"] = row["x_abs_p50"] = row["x_abs_p90"] = row["x_abs_p99"] = float("nan")
        row["x_kurtosis_raw"] = float("nan")

    Gact = (st.Gact / max(st.Cact, 1e-6)).float()
    Gsig = (st.Gsig / max(st.Csig, 1e-6)).float()
    Gn = (st.Gnoi / max(st.Cnoi, 1e-6)).float() if st.Gnoi is not None and st.Cnoi > 0 else None

    diag = torch.diagonal(Gact, dim1=1, dim2=2).cpu().numpy().reshape(-1)
    row["gact_diag_mean"] = float(np.mean(diag))
    row["gact_diag_cv"] = float(np.std(diag) / (np.mean(diag) + eps))
    row["gact_diag_p10"], row["gact_diag_p50"], row["gact_diag_p90"] = _safe_quantiles(diag, [0.1, 0.5, 0.9])

    off = Gact.clone()
    off[:, torch.arange(4), torch.arange(4)] = 0
    row["gact_offdiag_abs_mean"] = float(off.abs().mean().item())
    row["gact_offdiag_to_diag"] = float(off.abs().mean().item() / (torch.diagonal(Gact, dim1=1, dim2=2).mean().item() + eps))

    eye = torch.eye(4, dtype=Gact.dtype).unsqueeze(0)
    cond = torch.linalg.cond(Gact + 1e-6 * eye).cpu().numpy().reshape(-1)
    row["gact_cond_p50"], row["gact_cond_p90"], row["gact_cond_p99"] = _safe_quantiles(cond, [0.5, 0.9, 0.99])

    te = float(st.trend_energy.sum()) if st.trend_energy is not None else float("nan")
    se = float(st.season_energy.sum()) if st.season_energy is not None else float("nan")
    ne = float(st.noise_energy.sum()) if st.noise_energy is not None else float("nan")
    total_e = te + se + ne + 1e-9 if all(np.isfinite([te, se, ne])) else float("nan")
    row["trend_frac"] = te / total_e if np.isfinite(total_e) else float("nan")
    row["season_frac"] = se / total_e if np.isfinite(total_e) else float("nan")
    row["noise_frac"] = ne / total_e if np.isfinite(total_e) else float("nan")
    row["nsr_layer"] = ne / (te + se + 1e-9) if np.isfinite(te + se + ne) else float("nan")
    row["nsr_dataset_avg"] = float(getattr(st, "avg_nsr", float("nan")))

    pair_stats = _pair_score_summary(
        Wg.to(dtype=torch.float32),
        Gsig.to(dtype=torch.float32),
        Gact.to(dtype=torch.float32),
        Gn.to(dtype=torch.float32) if Gn is not None else None,
        eps=eps,
        ridge=ridge,
    )
    row.update(pair_stats)
    return row


def _print_ranked(rows, key, reverse=True, limit=10):
    ordered = sorted([r for r in rows if np.isfinite(r.get(key, np.nan))], key=lambda r: r[key], reverse=reverse)
    print(f"\nTop {min(limit, len(ordered))} layers by {key}:")
    for r in ordered[:limit]:
        print(
            f"  {r['layer']}: {key}={r[key]:.4f} "
            f"noise_frac={r.get('noise_frac', float('nan')):.3f} "
            f"snr_margin={r.get('snr_margin_p50', float('nan')):.3f} "
            f"obs_margin={r.get('obs_margin_p50', float('nan')):.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", default="OT")
    ap.add_argument("--train_end", type=int, required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--calib_select", choices=["first", "last"], default="last")
    ap.add_argument("--calib_windows", type=int, default=1024)
    ap.add_argument("--calib_batch", type=int, default=4)
    ap.add_argument("--max_calls_per_layer", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--error_power", type=float, default=0.0)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--include_quantile_head", type=int, default=0)
    ap.add_argument("--include_regex", default=".*")
    ap.add_argument("--out_dir", default="results/diagnostics")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    series = load_series(args.csv, args.col)
    X_train, Y_train = make_windows(series, 0, args.train_end, args.context, args.horizon, 1)
    X_pool = X_train[-args.calib_windows:] if args.calib_select == "last" else X_train[:args.calib_windows]
    Y_pool = Y_train[-args.calib_windows:] if args.calib_select == "last" else Y_train[:args.calib_windows]

    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    torch_mod = find_torch_module(tfm)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=args.horizon))

    preds_pool = []
    for i in range(0, len(X_pool), args.calib_batch):
        preds_pool.append(forecast_timesfm_point(tfm, X_pool[i:i + args.calib_batch], args.horizon))
    preds_pool = np.concatenate(preds_pool, axis=0)
    mse_pool, mae_pool = mse_mae(preds_pool, Y_pool)
    errs = np.mean((preds_pool - Y_pool) ** 2, axis=1)

    weights_sig = (errs / (errs.mean() + 1e-7)) ** args.error_power
    inv_err = (1.0 / (errs + 1e-7))
    weights_noi = (inv_err / (inv_err.mean() + 1e-7)) ** args.error_power

    targets = select_linears(torch_mod, bool(args.include_quantile_head), args.include_regex)
    stats = collect_stats(
        tfm,
        targets,
        X_pool,
        weights_sig.astype(np.float32),
        weights_noi.astype(np.float32),
        args.horizon,
        args.calib_batch,
        args.max_calls_per_layer,
    )

    nsrs = []
    for st in stats.values():
        if st.trend_energy is None or st.season_energy is None or st.noise_energy is None:
            continue
        te = float(st.trend_energy.sum())
        se = float(st.season_energy.sum())
        ne = float(st.noise_energy.sum())
        nsrs.append(ne / (te + se + 1e-9))
    avg_nsr = float(np.mean(nsrs)) if nsrs else float("nan")
    for st in stats.values():
        st.avg_nsr = avg_nsr

    rows = []
    for name, layer in targets:
        rows.append(_layer_distribution_row(name, layer, stats.get(name), ridge=args.ridge, eps=args.eps))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(args.csv).stem
    tag = args.tag or f"{stem}_h{args.horizon}_c{args.context}"
    out_csv = Path(args.out_dir) / f"{tag}_layer_stats.csv"
    out_json = Path(args.out_dir) / f"{tag}_summary.json"

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary = {
        "csv": args.csv,
        "context": args.context,
        "horizon": args.horizon,
        "calib_windows": len(X_pool),
        "calib_select": args.calib_select,
        "error_power": args.error_power,
        "pool_mse": float(mse_pool),
        "pool_mae": float(mae_pool),
        "err_mean": float(errs.mean()),
        "err_std": float(errs.std()),
        "err_p50": float(np.quantile(errs, 0.5)),
        "err_p90": float(np.quantile(errs, 0.9)),
        "err_p99": float(np.quantile(errs, 0.99)),
        "weights_sig_min": float(weights_sig.min()),
        "weights_sig_max": float(weights_sig.max()),
        "weights_sig_mean": float(weights_sig.mean()),
        "weights_noi_min": float(weights_noi.min()),
        "weights_noi_max": float(weights_noi.max()),
        "weights_noi_mean": float(weights_noi.mean()),
        "avg_nsr": avg_nsr,
        "num_targets": len(targets),
        "num_stats_layers": len(stats),
        "out_csv": str(out_csv),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    _print_ranked(rows, "gact_cond_p99", reverse=True, limit=10)
    _print_ranked(rows, "noise_frac", reverse=True, limit=10)
    _print_ranked(rows, "snr_winner_entropy", reverse=False, limit=10)
    print(f"\nSaved layer diagnostics to {out_csv}")
    print(f"Saved summary to {out_json}")


if __name__ == "__main__":
    main()
