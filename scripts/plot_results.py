import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def load_all_results(path):
    # data[dataset][context][horizon][method] = mse
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    try:
        if not os.path.exists(path):
            return data
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row['dataset']
                method = row['method']
                
                try:
                    h = int(row['horizon'])
                    c = int(row['context'])
                    mse = float(row['mse'])
                    data[ds][c][h][method] = mse
                except: continue
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return data

# Load all data from the centralized results file
results_path = 'restored_v13_sweep.csv'
all_data = load_all_results(results_path)

# Target directory for plots (artifacts dir)
target_dir = "/home/kevijayakumar/.gemini/antigravity/brain/b7246a56-dd6e-446b-84bf-4cba95bba77d"
os.makedirs(target_dir, exist_ok=True)

datasets = sorted(all_data.keys())
horizons = [96, 192, 336]
# Priority methods to show
preferred_methods = ['unified', 'wanda', 'sparsegpt', 'magnitude']
exclude_methods = {'unified_v9', 'unified_v12'}  # Legacy methods to hide
colors = {'unified': 'blue', 'wanda': 'purple', 'sparsegpt': 'green', 'magnitude': 'red'}
markers = {'unified': 'o', 'wanda': 'D', 'sparsegpt': 's', 'magnitude': '^'}

for ds in datasets:
    for c in sorted(all_data[ds].keys()):
        plt.figure(figsize=(10, 6))
        
        found_any = False
        # Get all methods present for this ds/context
        available_methods = set()
        for h in horizons:
            available_methods.update(all_data[ds][c][h].keys())
        
        # Plot in a specific order: preferred first, then others
        plot_order = [m for m in preferred_methods if m in available_methods]
        plot_order += [m for m in available_methods if m not in preferred_methods and m not in exclude_methods]

        for m in plot_order:
            x = []
            y = []
            for h in sorted(horizons):
                if m in all_data[ds][c][h]:
                    x.append(h)
                    y.append(all_data[ds][c][h][m])
            
            if x:
                color = colors.get(m, None)
                marker = markers.get(m, 'v')
                plt.plot(x, y, label=m, color=color, marker=marker, markersize=8, linewidth=2)
                found_any = True
        
        if not found_any:
            plt.close()
            continue

        plt.title(f"{ds} Performance Sweep (Context={c})", fontsize=14)
        plt.xlabel("Prediction Horizon (H)", fontsize=12)
        plt.ylabel("MSE (Lower is Better)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(horizons)
        
        fname = f"plot_v13_{ds}_C{c}.png"
        fpath = os.path.join(target_dir, fname)
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Generated {fname}")

print("Plotting Complete.")
