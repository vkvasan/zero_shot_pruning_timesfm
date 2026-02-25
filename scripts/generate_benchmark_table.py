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


def load_dense_csv(csv_path: Path):
    dense = {}
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return dense
    cols = set(rows[0].keys())
    for row in rows:
        if {"dataset", "horizon", "context", "mse"}.issubset(cols):
            key = (row["dataset"], int(row["horizon"]), int(row["context"]))
            dense[key] = float(row["mse"])
    return dense


METHOD_RANK = {m: i for i, m in enumerate(["dense"] + METHOD_ORDER)}


def rank_methods(values: dict, ordered_methods):
    return sorted(ordered_methods, key=lambda m: (values[m], METHOD_RANK[m]))


def fmt(v: float, bold: bool = False, underline: bool = False) -> str:
    s = f"{v:.6f}"
    if bold:
        return f"**{s}**"
    if underline:
        return f"<u>{s}</u>"
    return s


def build_markdown(cfg, dense_map=None):
    dense_map = dense_map or {}
    lines = []
    lines.append("# Benchmark Table (Best-Available Unified)")
    lines.append("")
    lines.append("- Source: `results/sweep_postpass_best_available.csv`")
    if dense_map:
        lines.append("- Includes `Dense` (no pruning) as a reference column.")
    lines.append("- Best value in each row is **bold**; second-best is <u>underlined</u>.")
    lines.append("")
    has_dense = bool(dense_map)
    header = ["Dataset", "H", "C"]
    if has_dense:
        header.append("Dense")
    header += ["Unified", "SparseGPT", "Wanda", "Magnitude", "Best"]
    lines.append("| " + " | ".join(header) + " |")
    aligns = ["---", "---:", "---:"]
    if has_dense:
        aligns.append("---:")
    aligns += ["---:", "---:", "---:", "---:", "---"]
    lines.append("|" + "|".join(aligns) + "|")
    for ds in DATASET_ORDER:
        for h in HORIZON_ORDER:
            for c in CONTEXT_ORDER:
                key = (ds, h, c)
                vals = cfg.get(key, {})
                if not all(m in vals for m in METHOD_ORDER):
                    continue
                row_vals = {m: vals[m] for m in METHOD_ORDER}
                ordered_methods = list(METHOD_ORDER)
                if has_dense and key in dense_map:
                    row_vals["dense"] = dense_map[key]
                    ordered_methods = ["dense"] + ordered_methods
                ranked = rank_methods(row_vals, ordered_methods)
                best = ranked[0]
                second = ranked[1]
                cells = [ds, str(h), str(c)]
                if has_dense and key in dense_map:
                    cells.append(fmt(dense_map[key], bold=(best == "dense"), underline=(second == "dense")))
                elif has_dense:
                    cells.append("")
                cells.extend(
                    [
                        fmt(vals["unified"], bold=(best == "unified"), underline=(second == "unified")),
                        fmt(vals["sparsegpt"], bold=(best == "sparsegpt"), underline=(second == "sparsegpt")),
                        fmt(vals["wanda"], bold=(best == "wanda"), underline=(second == "wanda")),
                        fmt(vals["magnitude"], bold=(best == "magnitude"), underline=(second == "magnitude")),
                        best,
                    ]
                )
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep_postpass_best_available.csv")
    ap.add_argument("--dense_csv", default="results/dense_baselines.csv")
    ap.add_argument("--out", default="results/benchmark_table_postpass_best_available.md")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    dense_csv_path = Path(args.dense_csv) if args.dense_csv else None
    out_path = Path(args.out)
    cfg = load_pivoted(csv_path)
    dense_map = {}
    if dense_csv_path and dense_csv_path.exists():
        dense_map = load_dense_csv(dense_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(cfg, dense_map))
    print(out_path)


if __name__ == "__main__":
    main()
