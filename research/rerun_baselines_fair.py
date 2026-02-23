#!/usr/bin/env python3
"""
Re-run ALL baseline configs with horizon-aware gram collection.
This produces a fair comparison by fixing the SparseGPT/Magnitude gram 
to use the actual target horizon instead of hardcoded H=96.
"""
import os, subprocess, csv

CONFIGS = [
    # dataset, horizon, context, train_end
    ("ETTm1", 96, 512, 49152),
    ("ETTm1", 96, 1024, 49152),
    ("ETTm1", 96, 2048, 49152),
    ("ETTm1", 192, 1024, 49152),
    ("ETTm1", 336, 1024, 49152),
    ("ETTm1", 720, 1024, 49152),
    ("ETTm2", 96, 512, 49152),
    ("ETTm2", 96, 1024, 49152),
    ("ETTm2", 96, 2048, 49152),
    ("ETTm2", 192, 1024, 49152),
    ("ETTm2", 336, 1024, 49152),
    ("ETTm2", 720, 1024, 49152),
    ("ETTh1", 96, 512, 8640),
    ("ETTh1", 96, 1024, 8640),
    ("ETTh1", 96, 2048, 8640),
    ("ETTh1", 192, 1024, 8640),
    ("ETTh1", 336, 1024, 8640),
    ("ETTh1", 720, 1024, 8640),
    ("ETTh2", 96, 512, 8640),
    ("ETTh2", 96, 1024, 8640),
    ("ETTh2", 96, 2048, 8640),
    ("ETTh2", 192, 1024, 8640),
    ("ETTh2", 336, 1024, 8640),
    ("ETTh2", 720, 1024, 8640),
]

METHODS = ["sparsegpt", "magnitude"]

RESULTS_FILE = "sweep_results_fair.csv"

def is_done(dataset, method, h, c):
    if not os.path.exists(RESULTS_FILE):
        return False
    with open(RESULTS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5 and row[0] == dataset and row[1] == method and int(row[2]) == h and int(row[3]) == c:
                return True
    return False

def main():
    # Write header if new
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["dataset", "method", "horizon", "context", "mse"])

    total = len(CONFIGS) * len(METHODS)
    done = 0
    for ds, h, c, train_end in CONFIGS:
        for mode in METHODS:
            done += 1
            if is_done(ds, mode, h, c):
                print(f"[{done}/{total}] SKIP {ds} {mode} H={h} C={c}")
                continue
            
            print(f"[{done}/{total}] Running {ds} {mode} H={h} C={c} ...")
            cmd = (
                f"micromamba run -n timesfm311 python -u baselines_2of4.py "
                f"--csv ETDataset/ETT-small/{ds}.csv --col OT "
                f"--train_end {train_end} --horizon {h} --context {c} "
                f"--mode {mode} --stride_test {h}"
            )
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = proc.stdout + proc.stderr
            
            # Parse MSE
            mse = None
            for line in output.split("\n"):
                tag = f"[{mode.upper()}] MSE="
                if tag in line:
                    mse = float(line.split("MSE=")[1].split()[0])
                    break
            
            if mse is not None:
                print(f"  > {mode}: MSE={mse:.4f}")
                with open(RESULTS_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([ds, mode, h, c, f"{mse:.6f}"])
            else:
                print(f"  > FAILED to parse MSE")
                print(output[-500:])

    print("\nDone! Results in", RESULTS_FILE)

if __name__ == "__main__":
    main()
