import pandas as pd
import numpy as np
import os
import sys
import subprocess

# Auto-install dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing matplotlib...")
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    print("Installing seaborn...")
    install("seaborn")
    import seaborn as sns

def plot_sweep():
    csv_file = "sweep_results.csv"
    if not os.path.exists(csv_file):
        print("sweep_results.csv not found.")
        return

    df = pd.read_csv(csv_file)
    
    # Filter methods to cleaner names
    # "unified" -> "Unified v2"
    # "sparsegpt" -> "SparseGPT"
    # "magnitude" -> "Magnitude"
    name_map = {
        "unified": "Unified v2",
        "sparsegpt": "SparseGPT",
        "magnitude": "Magnitude"
    }
    df["Method"] = df["method"].map(name_map)

    # We have two experimental axes:
    # 1. Horizon Sweep (where Context = 1024)
    # 2. Context Sweep (where Horizon = 96)
    
    datasets = df["dataset"].unique()
    sns.set_style("whitegrid")

    for dataset in datasets:
        ds_data = df[df["dataset"] == dataset]
        
        # --- Plot 1: MSE vs Horizon (fixed Context=1024) ---
        subset_h = ds_data[ds_data["context"] == 1024].sort_values("horizon")
        if not subset_h.empty and len(subset_h["horizon"].unique()) > 1:
            try:
                plt.figure(figsize=(6, 4))
                sns.lineplot(data=subset_h, x="horizon", y="mse", hue="Method", marker="o", palette="viridis")
                plt.title(f"{dataset} Horizon Sweep (Context=1024)", fontsize=12)
                plt.xlabel("Prediction Horizon (H)", fontsize=10)
                plt.ylabel("MSE (Lower is Better)", fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title='Method')
                plt.tight_layout()
                if not os.path.exists("plots"):
                    os.makedirs("plots")
                plt.savefig(f"plots/plot_{dataset}_horizon.png", dpi=150)
                plt.close()
                print(f"Saved plots/plot_{dataset}_horizon.png")
            except Exception as e:
                print(f"Could not plot horizon sweep for {dataset}: {e}")

        # --- Plot 2: MSE vs Context (fixed Horizon=96) ---
        subset_c = ds_data[ds_data["horizon"] == 96].sort_values("context")
        if not subset_c.empty and len(subset_c["context"].unique()) > 1:
            try:
                plt.figure(figsize=(6, 4))
                sns.lineplot(data=subset_c, x="context", y="mse", hue="Method", marker="s", palette="magma", linewidth=2.5)
                plt.title(f"{dataset} Context Sweep (Horizon=96)", fontsize=12)
                plt.xlabel("Context Length (C)", fontsize=10)
                plt.ylabel("MSE (Lower is Better)", fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title='Method')
                plt.tight_layout()
                if not os.path.exists("plots"):
                    os.makedirs("plots")
                plt.savefig(f"plots/plot_{dataset}_context.png", dpi=150)
                plt.close()
                print(f"Saved plots/plot_{dataset}_context.png")
            except Exception as e:
                print(f"Could not plot context sweep for {dataset}: {e}")

if __name__ == "__main__":
    plot_sweep()
