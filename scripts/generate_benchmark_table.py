import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASET_ORDER = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]
HORIZON_ORDER = [96, 192, 336]
CONTEXT_ORDER = [512, 1024, 2048]
METHOD_ORDER = ["unified", "sparsegpt", "wanda", "magnitude"]


def load_pivoted(csv_path: Path):
    cfg = defaultdict(dict)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], int(row["horizon"]), int(row["context"]))
            cfg[key][row["method"]] = float(row["mse"])
    return cfg


def winner_name(values: dict):
    return min(values.items(), key=lambda kv: kv[1])[0]


def fmt(v: float, bold: bool = False) -> str:
    s = f"{v:.6f}"
    return f"**{s}**" if bold else s


def build_markdown(cfg):
    lines = []
    lines.append("# Benchmark Table (Best-Available Unified)")
    lines.append("")
    lines.append("- Source: `results/sweep_postpass_best_available.csv`")
    lines.append("- `Unified` column is bolded for quick comparison.")
    lines.append("")
    lines.append("| Dataset | H | C | Unified | SparseGPT | Wanda | Magnitude | Best |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for ds in DATASET_ORDER:
        for h in HORIZON_ORDER:
            for c in CONTEXT_ORDER:
                key = (ds, h, c)
                vals = cfg.get(key, {})
                if not all(m in vals for m in METHOD_ORDER):
                    continue
                best = winner_name({m: vals[m] for m in METHOD_ORDER})
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            ds,
                            str(h),
                            str(c),
                            fmt(vals["unified"], bold=True),
                            fmt(vals["sparsegpt"]),
                            fmt(vals["wanda"]),
                            fmt(vals["magnitude"]),
                            best,
                        ]
                    )
                    + " |"
                )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep_postpass_best_available.csv")
    ap.add_argument("--out", default="results/benchmark_table_postpass_best_available.md")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    cfg = load_pivoted(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(cfg))
    print(out_path)


if __name__ == "__main__":
    main()
