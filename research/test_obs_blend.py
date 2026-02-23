#!/usr/bin/env python3
"""Test unified v3 (OBS-blended) on all key configs."""
import subprocess, csv, os

# All configs we care about  
CONFIGS = [
    # Losing to SparseGPT
    ("ETTm1", 336, 1024, 49152, 0),
    ("ETTm2", 336, 1024, 49152, 0.5),
    ("ETTm2", 720, 1024, 49152, 0.5),
    ("ETTh1", 336, 1024, 8640, 0),
    ("ETTh1", 96, 1024, 8640, 1.0),
    # Configs where we WIN (regression check)
    ("ETTm1", 96, 1024, 49152, 1.0),
    ("ETTm2", 96, 1024, 49152, 1.0),
    ("ETTh2", 96, 1024, 8640, 1.0),
    ("ETTh2", 720, 1024, 8640, 0),
]

RESULTS_FILE = "obs_test_results.csv"

BASE_CMD = (
    "micromamba run -n timesfm311 python -u snr_2of4_signal_noise_ratio2_v1.py "
    "--csv ETDataset/ETT-small/{ds}.csv --col OT "
    "--train_end {te} --horizon {h} --context {c} "
    "--score_mode unified --refit 1 --ridge 1e-5 "
    "--error_power {ep} --stride_test {h}"
)

# SparseGPT baselines for comparison
SPARSEGPT = {
    ("ETTm1", 336, 1024): 6.85,
    ("ETTm2", 336, 1024): 32.75,
    ("ETTm2", 720, 1024): 33.57,
    ("ETTh1", 336, 1024): 10.78,
    ("ETTh1", 96, 1024): 7.06,
    ("ETTm1", 96, 1024): 3.17,
    ("ETTm2", 96, 1024): 16.88,
    ("ETTh2", 96, 1024): 28.35,
    ("ETTh2", 720, 1024): 52.64,
}

def parse_mse(output):
    for line in output.split("\n"):
        if "[snr-2of4-refit] MSE=" in line:
            return float(line.split("MSE=")[1].split()[0])
    return None

def main():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["dataset","horizon","context","error_power","mse"])

    total = len(CONFIGS)
    for i, (ds, h, c, te, ep) in enumerate(CONFIGS):
        print(f"\n[{i+1}/{total}] {ds} H={h} C={c} P={ep}")
        cmd = BASE_CMD.format(ds=ds, te=te, h=h, c=c, ep=ep)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        mse = parse_mse(proc.stdout)
        
        # Print noise/beta info
        for line in proc.stdout.split("\n"):
            if "noise_frac" in line or "unified-v2" in line or "unified-v3" in line:
                print(f"  {line.strip()}")
        
        if mse is not None:
            sparse = SPARSEGPT.get((ds, h, c), None)
            marker = ""
            if sparse:
                if mse < sparse: marker = f" ✅ beats SparseGPT ({sparse:.2f})"
                else: marker = f" ❌ loses to SparseGPT ({sparse:.2f})"
            print(f"  -> MSE={mse:.4f}{marker}")
            with open(RESULTS_FILE, "a", newline="") as f:
                csv.writer(f).writerow([ds, h, c, ep, f"{mse:.6f}"])
        else:
            print(f"  -> FAILED")
            print(proc.stderr[-500:])

if __name__ == "__main__":
    main()
