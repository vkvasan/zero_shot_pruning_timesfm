# Unified v2 Pruning for Zero-Shot Time-Series (TimesFM)

This repository contains the implementation and experimental evaluation of a **pruning-time Competitive MoE** for 2:4 sparsity in TimesFM, with layer-wise expert selection across `Magnitude`, `Wanda`, `OBS`, and `SNR`-style pruning/reconstruction strategies.

## 🚀 Key Features

- **Pruning-Time MoE Routing**: Selects the pruning expert **per layer** (and `mask/refit` variant) instead of using one global pruning method.
- **Distribution-Aware Gating**: Uses activation / Gram statistics (e.g., NSR, conditioning, kurtosis) to bias routing and apply safe fallbacks.
- **Forecast-Aware Post-Pass**: Greedy multi-layer refinement on held-out windows to reduce forecast MSE after local pruning decisions.
- **Interaction Diagnostics**: Pairwise move diagnostics and pair-aware greedy hooks for analyzing non-additive layer interactions.
- **Benchmark + Plotting Tooling**: Full 144-config sweep support and updated plotting scripts/artifacts.

## 📈 Experimental Evidence

The branch results are tracked in a merged benchmark table built from `results/sweep_postpass_best_available.csv` (all-dataset post-pass sweep + targeted stronger ETTm2 reruns where available).

### Benchmark Table (Recommended View)

- Full 36-config table (all datasets × horizons × contexts), with **Unified** values highlighted:
  - `results/benchmark_table_postpass_best_available.md`
- Source CSV used for the table:
  - `results/sweep_postpass_best_available.csv`

### Optional Plots (Legacy View)

Line plots are still available in `results/plots/`, but the table above is the primary presentation for this branch.

## 📊 Performance Statistics (Best-Available Unified)
- **29 / 36 wins (80.6%)** vs **SparseGPT**
- **35 / 36 wins (97.2%)** vs **Wanda**
- **33 / 36 wins (91.7%)** vs **Magnitude**
- **25 / 36 best-overall** configurations across `{Unified, SparseGPT, Wanda, Magnitude}`
- Mean MSE across all 36 configs: **Unified = 16.837** vs **SparseGPT = 17.796**

## 🛠 Repository Structure

- `scripts/`:
  - `prune_unified.py`: Main pruning implementation (Competitive MoE + safety/post-pass logic).
  - `baselines.py`: Runner for Wanda, SparseGPT, and Magnitude baselines.
  - `run_sweep.py`: Orchestrator for full grid experimental sweeps.
  - `plot_results.py`: Legacy visualization script for sweep plots.
  - `generate_updated_graphs.py`: Merges updated results and regenerates repo-style plots.
  - `generate_benchmark_table.py`: Builds a single markdown benchmark table from merged results.
  - `generate_report.py`: Word document generator for technical specifications.
- `results/`: Sweep CSVs, merged comparison CSVs, and generated performance plots.
- `research/`: Legacy logs, draft implementations, and diagnostic tools.

## 🏁 Getting Started

### 1. Installation
Ensure you have a working `micromamba` or `conda` environment with Python 3.11.

```bash
# Clone the repository
git clone <repo_url>
cd zero_shot_pruning4

# Install dependencies
pip install -r requirements.txt
```

### 2. Running a Pruning Experiment
To prune a TimesFM model with the Unified MoE pruning logic:

```bash
micromamba run -n timesfm311 python scripts/prune_unified.py \
  --csv ETDataset/ETT-small/ETTm2.csv --col OT --train_end 49152 \
  --horizon 192 --context 1024 \
  --score_mode unified --refit 1 --ridge 1e-5 \
  --max_calls_per_layer 64 --calib_select last --nf_hi 0.0 --error_power 0 \
  --safe_policy rule_v1 \
  --greedy_postpass_k 16 --greedy_postpass_steps 6 \
  --greedy_postpass_eval_windows 32 --greedy_postpass_screen_k 12 \
  --greedy_postpass_screen_min_gain 0.03 --greedy_postpass_min_step_gain 0.02 \
  --stride_test 192
```

### 3. Executing the Full Grid Sweep
To run the baseline 144-config sweep:

```bash
python3 scripts/run_sweep.py
```

To run a `Unified`-only sweep with custom post-pass flags (used for the updated branch results):

```bash
SWEEP_METHODS=unified \
SWEEP_LOG_FILE=results/unified_postpass_all_fast.csv \
SWEEP_UNIFIED_EXTRA_FLAGS=" --max_calls_per_layer 64 --calib_select last --nf_hi 0.0 --error_power 0 --safe_policy rule_v1 --greedy_postpass_k 16 --greedy_postpass_steps 6 --greedy_postpass_eval_windows 32 --greedy_postpass_screen_k 12 --greedy_postpass_screen_min_gain 0.03 --greedy_postpass_min_step_gain 0.02 --greedy_postpass_max_cands_per_layer 8" \
python3 scripts/run_sweep.py
```

### 4. Regenerating Updated Graphs

```bash
micromamba run -n timesfm311 python scripts/generate_updated_graphs.py
```

### 5. Generating the Benchmark Table (Preferred Branch Presentation)

```bash
python3 scripts/generate_benchmark_table.py
```

*Key result files:* `results/sweep_postpass_best_available.csv`, `results/merged_unified_postpass_fast_vs_baselines.csv`, `results/ettm2_best_available_vs_baselines.csv`.
