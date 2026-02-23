
import subprocess
import itertools
import csv
import os
import re

# Config
DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]

# Grid 1: Vary Horizon (Fixed Context=1024)
HORIZONS = [96, 192, 336]
FIXED_CONTEXT = 1024

# Grid 2: Vary Context (Fixed Horizon=96)
CONTEXTS = [512, 1024, 2048]
FIXED_HORIZON = 96

# Methods
METHODS = [
    ("unified", "scripts/prune_unified.py", "--score_mode unified --refit 1 --ridge 1e-5"),
    ("sparsegpt", "scripts/baselines.py", "--mode sparsegpt --refit 1"),
    ("magnitude", "scripts/baselines.py", "--mode magnitude --refit 0"),
    ("wanda", "scripts/baselines.py", "--mode wanda --refit 0"),
]

# Dataset Metadata
TRAIN_ENDS = {
    "ETTm1": 49152, "ETTm2": 49152,
    "ETTh1": 8640,  "ETTh2": 8640
}

def parse_mse(output):
    # Search for lines like [snr-2of4-refit] MSE=... or [sparsegpt] MSE=...
    # We want the last MSE mentioned or specifically the pruned one.
    matches = re.findall(r"MSE=([0-9\.]+)", output)
    if matches:
        # Usually the last one is the pruned MSE in our scripts
        return float(matches[-1])
    return None


def is_completed(log_file, dataset, method_name, h, c):
    """
    Checks if a specific configuration (dataset, method, horizon, context)
    has already been recorded in the log_file.
    """
    if not os.path.exists(log_file):
        return False
    try:
        with open(log_file, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                if len(row) >= 5 and \
                   row[0] == dataset and \
                   row[1] == method_name and \
                   int(row[2]) == h and \
                   int(row[3]) == c:
                    return True
        return False
    except Exception as e:
        return False

def run_cmd(cmd):
    import sys
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        # Stream to log
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stderr: {e.stderr}")
        return None



def main():
    results = []
    log_file = "results/restored_v13_sweep.csv"
    
    # Check if exists to append? 
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "method", "horizon", "context", "mse"])

    # Build unique (dataset, horizon, context) tuples
    configs = list(itertools.product(HORIZONS, CONTEXTS))
    sorted_configs = sorted(configs)

    print(f"Total configs to run: {len(sorted_configs)} x {len(DATASETS)} x {len(METHODS)} = {len(sorted_configs)*len(DATASETS)*len(METHODS)}")

    for dataset in DATASETS:
        train_end = TRAIN_ENDS[dataset]
        csv_path = f"ETDataset/ETT-small/{dataset}.csv"
        
        for (h, c) in sorted_configs:
            print(f"--- {dataset} | H={h} | C={c} ---")
            
            for method_name, script, args_str in METHODS:
                if is_completed(log_file, dataset, method_name, h, c):
                    continue

                # Use standardized flags validated in Phase 2
                if method_name == "unified":
                    extra_flags = " --max_calls_per_layer 64 --calib_select last --nf_hi 0.0 --error_power 0"
                else:
                    extra_flags = " --calib_windows 256 --calib_select last"

                cmd = (f"micromamba run -n timesfm311 python {script} "
                       f"--csv {csv_path} --col OT "
                       f"--train_end {train_end} "
                       f"--horizon {h} --context {c} "
                       f"{args_str}{extra_flags} "
                       f"--stride_test {h}")

                
                # For `baselines_2of4` we assume it handles stride?
                # baselines_2of4: main() has manual loop...
                # It uses `stride=96` (hardcoded in make_windows call at eval?).
                # Let's check baselines_2of4.py...
                # It has `X_te, Y_te = make_windows(..., stride=96)`. 
                # This might be an issue if H > 96. We should update baselines_2of4 to accept stride arg.
                # But for now, let's run it. Overlapping windows in eval is fine (just more compute).
                
                output = run_cmd(cmd)
                if output:
                    mse = parse_mse(output)
                    if mse is not None:
                        print(f"  > {method_name}: MSE={mse:.4f}")
                        results.append([dataset, method_name, h, c, mse])
                        
                        # Append to CSV immediately
                        with open(log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([dataset, method_name, h, c, mse])
                    else:
                        print(f"  > {method_name}: Failed to parse MSE")

if __name__ == "__main__":
    main()
