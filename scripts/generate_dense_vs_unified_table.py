import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASET_ORDER = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]
HORIZON_ORDER = [96, 192, 336]
CONTEXT_ORDER = [512, 1024, 2048]


def load_dense(csv_path: Path):
    dense = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], int(row["horizon"]), int(row["context"]))
            dense[key] = float(row["mse"])
    return dense


def load_unified(csv_path: Path):
    unified = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["method"] != "unified":
                continue
            key = (row["dataset"], int(row["horizon"]), int(row["context"]))
            unified[key] = float(row["mse"])
    return unified


def fmt(v: float, bold: bool = False, underline: bool = False) -> str:
    s = f"{v:.6f}"
    if bold:
        return f"**{s}**"
    if underline:
        return f"<u>{s}</u>"
    return s


def build_markdown(dense_map, unified_map):
    lines = []
    lines.append("# Dense vs Unified (Best-Available Unified)")
    lines.append("")
    lines.append("- Dense source: `results/dense_baselines.csv`")
    lines.append("- Unified source: `results/sweep_postpass_best_available.csv` (method=`unified`)")
    lines.append("- Best value in each row is **bold**; second-best is <u>underlined</u>.")
    lines.append("")
    lines.append("| Dataset | H | C | Dense | Unified | Delta (U-D) | Better |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for ds in DATASET_ORDER:
        for h in HORIZON_ORDER:
            for c in CONTEXT_ORDER:
                key = (ds, h, c)
                if key not in dense_map or key not in unified_map:
                    continue
                dense = dense_map[key]
                unified = unified_map[key]
                best_is_unified = unified < dense
                cells = [
                    ds,
                    str(h),
                    str(c),
                    fmt(dense, bold=not best_is_unified, underline=best_is_unified),
                    fmt(unified, bold=best_is_unified, underline=not best_is_unified),
                    f"{(unified - dense):+.6f}",
                    "unified" if best_is_unified else "dense",
                ]
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_csv", default="results/dense_baselines.csv")
    ap.add_argument("--sweep_csv", default="results/sweep_postpass_best_available.csv")
    ap.add_argument("--out", default="results/dense_vs_unified_table_postpass_best_available.md")
    args = ap.parse_args()

    dense_map = load_dense(Path(args.dense_csv))
    unified_map = load_unified(Path(args.sweep_csv))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(dense_map, unified_map))
    print(out_path)


if __name__ == "__main__":
    main()
