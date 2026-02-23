#!/usr/bin/env python3
"""
Test temporal proximity calibration (--calib_select last) vs old (first)
on the configs where unified loses to SparseGPT.
"""
import subprocess, csv, os

CONFIGS = [
    # dataset, h, c, train_end, current_error_power
    ("ETTm1", 336, 1024, 49152, 0),
    ("ETTm2", 336, 1024, 49152, 0.5),
    ("ETTm2", 720, 1024, 49152, 0.5),
    ("ETTh1", 336, 1024, 8640, 0),
    ("ETTh1", 96, 1024, 8640, 1.0),
    # Also test configs where we already win to make sure we don't regress
    ("ETTm1", 96, 1024, 49152, 1.0),
    ("ETTm2", 96, 1024, 49152, 1.0),
    ("ETTh2", 720, 1024, 8640, 0),
]

RESULTS_FILE = "temporal_proximity_test.csv"

BASE_CMD = (
    "micromamba run -n timesfm311 python -u snr_2of4_signal_noise_ratio2_v1.py "
    "--csv ETDataset/ETT-small/{ds}.csv --col OT "
    "--train_end {te} --horizon {h} --context {c} "
    "--score_mode unified --refit 1 --ridge 1e-5 "
    "--error_power {ep} --stride_test {h} "
    "--calib_select {sel}"
)

def parse_mse(output):
    for line in output.split("\n"):
        if "[snr-2of4-refit] MSE=" in line:
            return float(line.split("MSE=")[1].split()[0])
    return None

def main():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["dataset","horizon","context","calib_select","error_power","mse"])

    total = len(CONFIGS) * 2
    done = 0
    for ds, h, c, te, ep in CONFIGS:
        for sel in ["first", "last"]:
            done += 1
            print(f"[{done}/{total}] {ds} H={h} C={c} calib_select={sel} P={ep}")
            cmd = BASE_CMD.format(ds=ds, te=te, h=h, c=c, ep=ep, sel=sel)
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            mse = parse_mse(proc.stdout)
            if mse is not None:
                print(f"  -> MSE={mse:.4f}")
                with open(RESULTS_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([ds, h, c, sel, ep, f"{mse:.6f}"])
            else:
                print(f"  -> FAILED")
                print(proc.stdout[-300:])
                print(proc.stderr[-300:])

    # Summary
    print("\n=== SUMMARY ===")
    from collections import defaultdict
    results = defaultdict(dict)
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["dataset"], int(r["horizon"]), int(r["context"]))
            results[key][r["calib_select"]] = float(r["mse"])
    
    print(f"{'Config':<25} {'first':>10} {'last':>10} {'Δ':>10} {'Δ%':>8}")
    print("-" * 65)
    for key in sorted(results.keys()):
        d = results[key]
        if "first" in d and "last" in d:
            delta = d["last"] - d["first"]
            pct = delta / d["first"] * 100
            marker = "✅" if delta < 0 else "❌"
            print(f"{key[0]} H={key[1]} C={key[2]:<5} {d['first']:>10.4f} {d['last']:>10.4f} {delta:>+10.4f} {pct:>+7.1f}% {marker}")

if __name__ == "__main__":
    main()
