import subprocess
import csv
import os
import re

datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]
horizons = [96, 192, 336]
contexts = [512, 1024, 2048]

results_file = "sweep_results.csv"
fieldnames = ["dataset", "method", "horizon", "context", "mse"]

def is_done(ds, method, h, c):
    if not os.path.exists(results_file):
        return False
    try:
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['dataset'] == ds and row['method'] == method and \
                   int(row['horizon']) == h and int(row['context']) == c:
                    return True
    except: pass
    return False

if not os.path.exists(results_file):
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

train_ends = {
    "ETTm1": 49152, "ETTm2": 49152,
    "ETTh1": 8640,  "ETTh2": 8640
}

# Methods to run
methods = [
    ("unified_v12", "snr_2of4_signal_noise_ratio2_v1.py", "--score_mode unified --refit 1"),
    ("sparsegpt", "baselines_2of4.py", "--mode sparsegpt --refit 1"),
    ("magnitude", "baselines_2of4.py", "--mode magnitude --refit 0")
]

for ds in datasets:
    train_end = train_ends[ds]
    csv_path = f"ETDataset/ETT-small/{ds}.csv"
    for h in horizons:
        for c in contexts:
            for m_name, script, extra in methods:
                if is_done(ds, m_name, h, c):
                    # print(f"Skipping: {ds} {m_name} H={h} C={c} (already done)", flush=True)
                    continue
                    
                print(f"Running: {ds} {m_name} H={h} C={c}...", flush=True)
                cmd = f"micromamba run -n timesfm311 python {script} --csv {csv_path} --train_end {train_end} --horizon {h} --context {c} {extra}"
                try:
                    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
                    
                    # Search specifically for the final pruned MSE
                    # Unified: [snr-2of4-refit] MSE=...
                    # SparseGPT: [SPARSEGPT] MSE=...
                    # Magnitude: [MAGNITUDE] MSE=...
                    mse = None
                    # Search from bottom up for the last "MSE="
                    for line in reversed(output.split('\n')):
                        if "MSE=" in line and ("refit" in line.lower() or "SPARSEGPT" in line or "MAGNITUDE" in line):
                            match = re.search(r"MSE=([0-9\.]+)", line)
                            if match:
                                mse = float(match.group(1))
                                break
                    
                    if mse is not None:
                        with open(results_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow({"dataset": ds, "method": m_name, "horizon": h, "context": c, "mse": mse})
                        print(f"  -> MSE={mse}", flush=True)
                    else:
                        print(f"  -> ERROR: MSE not found in output of {ds} {m_name} H={h} C={c}", flush=True)
                        # Optional: write log if error
                except Exception as e:
                    print(f"  -> FAILED: {e}", flush=True)

print("Full Grid Search Complete.", flush=True)
