
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    if not os.path.exists("sweep_results.csv"):
        print("sweep_results.csv not found.")
        return

    df = pd.read_csv("sweep_results.csv")
    
    # Clean up names for better legend
    method_map = {
        "unified": "Unified MoE (Ours)",
        "sparsegpt": "SparseGPT",
        "magnitude": "Magnitude Pruning"
    }
    df["method"] = df["method"].map(lambda x: method_map.get(x, x))

    datasets = df["dataset"].unique()
    
    # Create directory for plots if not exists
    os.makedirs("plots", exist_ok=True)

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        
        # Plot 1: MSE vs Horizon (Fixed Context=1024)
        plt.figure(figsize=(10, 6))
        sub_df = ds_df[ds_df["context"] == 1024]
        if not sub_df.empty:
            sns.lineplot(data=sub_df, x="horizon", y="mse", hue="method", marker="o", linewidth=2.5)
            plt.title(f"MSE vs Horizon (Context=1024) - {ds}", fontsize=14)
            plt.ylabel("MSE", fontsize=12)
            plt.xlabel("Horizon (H)", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(f"plots/{ds}_mse_vs_horizon.png", dpi=200)
            plt.close()

        # Plot 2: MSE vs Context (Fixed Horizon=336 or max available)
        plt.figure(figsize=(10, 6))
        # Use H=336 if available, else 96
        target_h = 336 if 336 in ds_df["horizon"].values else 96
        sub_df = ds_df[ds_df["horizon"] == target_h]
        if not sub_df.empty:
            sns.lineplot(data=sub_df, x="context", y="mse", hue="method", marker="s", linewidth=2.5)
            plt.title(f"MSE vs Context (Horizon={target_h}) - {ds}", fontsize=14)
            plt.ylabel("MSE", fontsize=12)
            plt.xlabel("Context Length (C)", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(f"plots/{ds}_mse_vs_context.png", dpi=200)
            plt.close()

    print("Plots generated in 'plots/' directory.")

if __name__ == "__main__":
    main()
