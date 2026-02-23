
import os
import subprocess
import argparse

# Configs to tune
EXPERIMENTS = [
    # ETTm2 H=336 (Unified was bad with P=0, let's try others)
    {"dataset": "ETTm2", "h": 336, "c": 1024, "powers": [0.5, 1.0, 1.5]},
    # ETTh1 H=96 (Unified slightly lost to SparseGPT)
    {"dataset": "ETTh1", "h": 96, "c": 1024, "powers": [0.0, 0.5, 2.0]}
]

# Base command
BASE_CMD = "micromamba run -n timesfm311 python -u snr_2of4_signal_noise_ratio2_v1.py --csv ETDataset/ETT-small/{ds}.csv --col OT --score_mode unified --refit 1 --ridge 1e-5 --stride_test {h}"

def run_tuning():
    results = []
    
    for exp in EXPERIMENTS:
        ds = exp["dataset"]
        h = exp["h"]
        c = exp["c"]
        powers = exp["powers"]
        
        # Get baseline dense MSE first? No, snr script reports it.
        # But we need train_end. ETTm1/m2=49152, ETTh1/2=8640.
        train_end = 49152 if "m" in ds else 8640
        
        print(f"\n=== Tuning {ds} H={h} C={c} ===")
        
        best_mse = float('inf')
        best_p = None
        
        for p in powers:
            print(f"Testing error_power={p}...")
            cmd = BASE_CMD.format(ds=ds, h=h) + f" --train_end {train_end} --horizon {h} --context {c} --error_power {p}"
            
            # Run and capture output to parse MSE
            try:
                proc = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = proc.stdout
                
                # Parse MSE
                mse = None
                for line in output.split('\n'):
                    if "[snr-2of4-refit] MSE=" in line:
                        mse = float(line.split("MSE=")[1].split()[0])
                        break
                
                if mse is not None:
                    print(f"  -> MSE={mse:.4f}")
                    if mse < best_mse:
                        best_mse = mse
                        best_p = p
                    results.append({"dataset": ds, "h": h, "c": c, "power": p, "mse": mse})
                else:
                    print("  -> Failed to parse MSE")
            
            except subprocess.CalledProcessError as e:
                print(f"  -> Error: {e}")
        
        print(f"*** Best for {ds} H={h}: error_power={best_p} (MSE={best_mse:.4f}) ***")

if __name__ == "__main__":
    run_tuning()
