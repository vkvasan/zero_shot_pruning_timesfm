# Unified v2 Pruning for Zero-Shot Time-Series (TimesFM)

This repository contains the implementation and experimental verification of **Unified v13 Pruning**, a state-of-the-art methodology for 2:4 sparsity in zero-shot foundation models (TimesFM).

## 🚀 Key Features

- **Unified v13 Logic**: A robust Competitive MoE strategy that selects the optimal reconstructor (Wanda, OBS, Magnitude, or SNR-weighted) per layer based on local validation reconstructions.
- **Foundational Generalization**: Specifically optimized for long-horizon and seasonal datasets where standard pruning methods (SparseGPT, Wanda) often fail.
- **Comprehensive Benchmarks**: Tools to execute a full 144-configuration Grid (Unified vs Wanda vs SparseGPT vs Magnitude) across all ETT datasets.
- **Automated Reporting**: Visual and document-based result synthesis.

## 📊 Performance Statistics

Unified v13 consistently outperforms traditional baselines across diverse benchmarks:
- **83% Win Rate** against Wanda.
- **58% Win Rate** against SparseGPT.
- **23% MSE Improvement** on ETTm1 (H=336, C=2048) by preventing refit-driven hallucinations.

## 🛠 Repository Structure

- `scripts/`:
  - `prune_unified.py`: Main pruning implementation using Unified v13 logic.
  - `baselines.py`: Runner for Wanda, SparseGPT, and Magnitude baselines.
  - `run_sweep.py`: Orchestrator for full grid experimental sweeps.
  - `plot_results.py`: Visualization engine for MSE vs Horizon/Context.
  - `generate_report.py`: Word document generator for technical specifications.
- `results/`: Contains `restored_v13_sweep.csv` and generated performance plots.
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
To prune a TimesFM model using the Unified v13 method:

```bash
python scripts/prune_unified.py \
    --dataset ETTm1 \
    --horizon 96 \
    --context 1024 \
    --sparsity 2:4
```

### 3. Executing the Full Grid Sweep
To replicate the full experimental evidence reported in the technical brief:

```bash
python scripts/run_sweep.py --baseline Unified
```

### 4. Generating Reports
To generate the final technical MS Word report:

```bash
python scripts/generate_report.py
```

## 📜 Methodology
Unified v13 utilizes an **Error-Weighted Gram Matrix** and a **Spectral Noise Guard**. It adaptively switches to an SNR-weighted expert in high-noise seasonal regimes (E.g., ETTm1) while maintaining the precision of Optimal Brain Surgeon (OBS) on clean signal datasets (E.g., ETTh2).

---
*For detailed experimental results, refer to `results/restored_v13_sweep.csv`.*
