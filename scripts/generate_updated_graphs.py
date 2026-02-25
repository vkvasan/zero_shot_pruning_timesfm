import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]
HORIZONS = [96, 192, 336]
CONTEXTS = [512, 1024, 2048]

METHOD_ORDER = ["unified", "sparsegpt", "wanda", "magnitude"]
METHOD_LABELS = {
    "unified": "Unified",
    "sparsegpt": "SparseGPT",
    "wanda": "Wanda",
    "magnitude": "Magnitude",
}
COLORS = {
    "unified": "#1f77b4",
    "sparsegpt": "#2ca02c",
    "wanda": "#9467bd",
    "magnitude": "#d62728",
}
MARKERS = {
    "unified": "o",
    "sparsegpt": "s",
    "wanda": "D",
    "magnitude": "^",
}


def read_csv_rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def build_best_available_unified_map(
    unified_fast_path: Path, ettm2_best_path: Path | None
):
    unified_map: dict[tuple[str, int, int], float] = {}
    for row in read_csv_rows(unified_fast_path):
        key = (row["dataset"], int(row["horizon"]), int(row["context"]))
        unified_map[key] = float(row["mse"])

    if ettm2_best_path and ettm2_best_path.exists():
        for row in read_csv_rows(ettm2_best_path):
            key = (row["dataset"], int(row["horizon"]), int(row["context"]))
            unified_map[key] = float(row["unified_best_available"])

    return unified_map


def write_merged_sweep(
    base_sweep_path: Path, unified_map: dict[tuple[str, int, int], float], out_path: Path
):
    rows = []
    for row in read_csv_rows(base_sweep_path):
        ds = row["dataset"]
        method = row["method"]
        h = int(row["horizon"])
        c = int(row["context"])
        mse = float(row["mse"])
        if method == "unified":
            mse = unified_map[(ds, h, c)]
        rows.append(
            {
                "dataset": ds,
                "method": method,
                "horizon": h,
                "context": c,
                "mse": mse,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "horizon", "context", "mse"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "horizon": row["horizon"],
                    "context": row["context"],
                    "mse": f"{row['mse']:.6f}",
                }
            )
    return rows


def nested_results(rows):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        data[row["dataset"]][int(row["context"])][int(row["horizon"])][row["method"]] = float(
            row["mse"]
        )
    return data


def plot_series(ax, xs, ys, method):
    ax.plot(
        xs,
        ys,
        label=METHOD_LABELS.get(method, method),
        color=COLORS.get(method),
        marker=MARKERS.get(method, "o"),
        linewidth=2.2,
        markersize=7,
    )


def plot_repo_style(data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if ds not in data:
            continue

        c = 1024
        plt.figure(figsize=(6, 4))
        found = False
        for method in METHOD_ORDER:
            xs, ys = [], []
            for h in HORIZONS:
                if method in data[ds][c][h]:
                    xs.append(h)
                    ys.append(data[ds][c][h][method])
            if xs:
                plot_series(plt.gca(), xs, ys, method)
                found = True
        if found:
            plt.title(f"{ds} Horizon Sweep (Context=1024)", fontsize=12)
            plt.xlabel("Prediction Horizon (H)", fontsize=10)
            plt.ylabel("MSE (Lower is Better)", fontsize=10)
            plt.xticks(HORIZONS)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(title="Method", fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f"plot_{ds}_horizon.png", dpi=150)
        plt.close()

        h = 96
        plt.figure(figsize=(6, 4))
        found = False
        for method in METHOD_ORDER:
            xs, ys = [], []
            for c in CONTEXTS:
                if method in data[ds][c][h]:
                    xs.append(c)
                    ys.append(data[ds][c][h][method])
            if xs:
                plot_series(plt.gca(), xs, ys, method)
                found = True
        if found:
            plt.title(f"{ds} Context Sweep (Horizon=96)", fontsize=12)
            plt.xlabel("Context Length (C)", fontsize=10)
            plt.ylabel("MSE (Lower is Better)", fontsize=10)
            plt.xticks(CONTEXTS)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(title="Method", fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f"plot_{ds}_context.png", dpi=150)
        plt.close()


def plot_alt_style(data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if ds not in data:
            continue

        plt.figure(figsize=(10, 6))
        found = False
        for method in METHOD_ORDER:
            xs, ys = [], []
            for h in HORIZONS:
                if method in data[ds][1024][h]:
                    xs.append(h)
                    ys.append(data[ds][1024][h][method])
            if xs:
                plot_series(plt.gca(), xs, ys, method)
                found = True
        if found:
            plt.title(f"MSE vs Horizon (Context=1024) - {ds}", fontsize=14)
            plt.xlabel("Horizon (H)", fontsize=12)
            plt.ylabel("MSE", fontsize=12)
            plt.xticks(HORIZONS)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / f"{ds}_mse_vs_horizon.png", dpi=200)
        plt.close()

        target_h = 336 if 336 in data[ds][512] or 336 in data[ds][1024] or 336 in data[ds][2048] else 96
        plt.figure(figsize=(10, 6))
        found = False
        for method in METHOD_ORDER:
            xs, ys = [], []
            for c in CONTEXTS:
                if method in data[ds][c][target_h]:
                    xs.append(c)
                    ys.append(data[ds][c][target_h][method])
            if xs:
                plot_series(plt.gca(), xs, ys, method)
                found = True
        if found:
            plt.title(f"MSE vs Context (Horizon={target_h}) - {ds}", fontsize=14)
            plt.xlabel("Context Length (C)", fontsize=12)
            plt.ylabel("MSE", fontsize=12)
            plt.xticks(CONTEXTS)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / f"{ds}_mse_vs_context.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_sweep", default="results/restored_v13_sweep.csv")
    ap.add_argument("--unified_fast", default="results/unified_postpass_all_fast.csv")
    ap.add_argument("--ettm2_best", default="results/ettm2_best_available_vs_baselines.csv")
    ap.add_argument("--merged_out", default="results/sweep_postpass_best_available.csv")
    ap.add_argument("--plots_out", default="results/plots_postpass_best_available")
    args = ap.parse_args()

    base_sweep_path = Path(args.base_sweep)
    unified_fast_path = Path(args.unified_fast)
    ettm2_best_path = Path(args.ettm2_best) if args.ettm2_best else None
    merged_out = Path(args.merged_out)
    plots_out = Path(args.plots_out)

    unified_map = build_best_available_unified_map(unified_fast_path, ettm2_best_path)
    merged_rows = write_merged_sweep(base_sweep_path, unified_map, merged_out)
    data = nested_results(merged_rows)
    plot_repo_style(data, plots_out)
    plot_alt_style(data, plots_out)

    print(f"[merged] {merged_out}")
    print(f"[plots] {plots_out}")


if __name__ == "__main__":
    main()
