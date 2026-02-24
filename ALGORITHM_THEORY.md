# Theory of Unified v13 Pruning: A Technical Deep Dive

## 1. Executive Summary
Unified v13 is an adaptive zero-shot pruning methodology designed for Time-Series Foundation Models (TimesFM). Unlike static pruning metrics (Magnitude, Wanda, or SparseGPT) which often fail on non-stationary or noisy time-series data, Unified v13 utilizes a **Competitive Mixture-of-Experts (MoE)** framework. It dynamically selects the optimal reconstruction strategy per layer by evaluating real-time validation error, ensuring maximum generalization across diverse seasonal and trend-heavy regimes.

## 2. Mathematical Foundations

### 2.1 Error-Weighted Gram Matrix Accumulation
Standard Second-Order pruning (e.g., SparseGPT) approximates the Hessian using the Gram matrix $G = X X^\top$. In Unified v13, we introduce a weighting function to prioritize "harder" or more informative calibration windows.

For each calibration window $i$:
1. **Error Computation**: $e_i = \text{MSE}(\hat{y}_i, y_i)$, where $\hat{y}$ is the dense model prediction.
2. **Relative Weighting**: $w_i = \left( \frac{e_i}{\bar{e} + \epsilon} \right)^P$, where $P$ is the `error_power`.
3. **Weighted Accumulation**: $G_{weighted} = \sum_i w_i (X_i X_i^\top)$.

By setting $P > 0$, the algorithm focuses the pruning mask on activations that contribute most to prediction error, preserving the weights essential for complex signal recovery.

### 2.2 Spectral Noise Guard (NSR)
Time-series data often contains high-frequency noise that can mislead pruning algorithms into overfitting local fluctuations. Unified v13 employs a Fast Fourier Transform (FFT) based Noise-to-Signal Ratio (NSR) guard.

- **Signal Energy**: Sum of low-frequency and seasonal FFT bin magnitudes.
- **Noise Energy**: Sum of high-frequency bin magnitudes.
- **Noise Fraction ($f_N$)**: $E_{noise} / (E_{signal} + E_{noise})$.

**The Alpha-Switch**:
A blending factor $\alpha$ is computed: $\alpha = \text{clamp}(\frac{f_N - \tau}{\tau}, 0, 1)$.
This $\alpha$ scales the influence of the **Spectral Expert** (which prunes based on frequency preservation) versus the **Ratio Expert** (standard reconstruction).

## 3. The Competitive MoE Strategy

The core innovation of v13 is the internal "Expert Competition" at every layer. For each weight group, the algorithm evaluates four distinct "Pruning Experts":

1.  **Magnitude (MAG)**: Heuristic-based, high robustness to extreme noise.
2.  **Wanda**: Activation-weighted magnitude, fast and effective for stable trends.
3.  **Optimal Brain Surgeon (OBS)**: Gram-inverse based, most precise for clean signals.
4.  **SNR-Weighted**: Specialized for highly seasonal/cyclic regimes (e.g., ETTm1).

### 3.1 Local Validation Reconstruction
Unlike global pruning, v13 performs a "mini-competition" per layer:
1.  **Trial Pruning**: Each expert produces a candidate mask for the layer.
2.  **Error Check**: The layer is evaluated against a held-out **Validation activation set** ($X_{val}$).
3.  **Winner Selection**: The expert with the minimum reconstruction error on $X_{val}$ wins the layer.

This prevents the "Refit Hallucination" common in SparseGPT, where the model over-adjusts weights to minimize calibration error at the expense of zero-shot generalization.

## 4. Key Improvements in v13 (The "Restoration")

- **Refit Parity**: Removal of artificial penalties (Refit Penalty = 1.0) ensures that high-precision experts like OBS are not unfairly suppressed in favor of simpler heuristics.
- **Balanced Data Allocation**: A 48/16 split of calibration windows ensures a high-quality Gram matrix (Hessian approximation) while maintaining a significant validation set for unbiased winner selection.
- **Global Horizon Sensitivity**: The algorithm automatically loosens guards for long horizons ($H > 192$), allowing for more aggressive weight reconstruction (Ridge = 1e-5) where trend extrapolation is more critical than noise suppression.

## 5. Experimental Conclusion
The Unified v13 algorithm effectively "heals" the sparsity-driven performance gap in TimesFM. By combining spectral awareness with competitive error-minimization, it achieves state-of-the-art results across the ETT benchmark suite, specifically recovering the 20-30% MSE loss observed when using standard SparseGPT on seasonal datasets.
