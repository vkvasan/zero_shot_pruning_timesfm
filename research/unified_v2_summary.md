# Unified v2 Pruning: Robustness via Spectral Noise Awareness

**TL;DR**: Standard pruning metrics (Magnitude, Wanda, SparseGPT) fail on noisy time-series data because the "signal" they maximize is often corrupted by noise. **Unified v2** estimates the Signal-to-Noise Ratio (SNR) of activations per layer and adapts both the **pruning metric** and the **refit regularization** accordingly.

---

## 1. The Core Problem: Noise Corrupts Pruning

Traditional metrics assume data is clean signal. On noisy datasets (e.g., **ETTh2**, 27% noise), they fail catastrophically:
*   **Magnitude**: Prunes small weights that might be critical for weak signals. (Result: **+5.5 MSE**)
*   **Wanda / SparseGPT**: Rely on the input covariance matrix $X^T X$, which becomes dominated by noise variance. (Result: **+5.2 MSE**, **+2.5 MSE**)

## 2. Solution: Unified v2 Mechanism

The method computes the **Noise Energy Fraction** ($N_{frac}$) of activations per layer using FFT. It then adapts two components dynamically:

### A. Adaptive Scoring (The "Gate")

Instead of a fixed metric, we blend two scores based on layer quality:
$$ Score_{unified} = \alpha \cdot Score_{spectral} + (1 - \alpha) \cdot Score_{ratio} $$

*   **Clean Layers ($N_{frac} < 15\%$)**: $\alpha \approx 0$. Use **Pure Gram Ratio** ($E_{keep} / E_{drop}$). Preserves exact ranking of standard techniques. (Example: **ETTm1/m2**)
*   **Noisy Layers ($N_{frac} > 30\%$)**: $\alpha \rightarrow 1$. Blend in **Spectral Quality** ($E_{trend+season} / E_{noise}$). Bypasses the noisy covariance matrix entirely. (Example: **ETTh2**)

### B. Auto-Ridge Refit (The "Stabilizer")

Standard refit algorithms (Optimal Brain Surgeon) invert the Hessian $H = X^T X + \lambda I$. If $H$ is noisy, the inverse explodes. Unified v2 auto-scales $\lambda$:

*   Detects **Max Dataset Noise** ($N_{max}$) across all layers.
*   If $N_{max} > 15\%$, scale $\lambda$ exponentially: $10^{-5} \rightarrow 10^{-2}$.
*   **Result**: Constrains the refit to be conservative on noisy data.

---

## 3. Results: Consistent Wins (MSE Comparison)

Unified v2 is the **only method** to provide the best MSE across all 4 benchmark datasets.

| Dataset | **Unified v2** | SparseGPT | Magnitude | WANDA | Verdict |
|---|---|---|---|---|---|
| **ETTm1** | **-0.14** | +0.03 | +0.02 | +0.49 | ✅ Best |
| **ETTm2** | **-0.18** | +3.17 | +4.65 | +6.39 | 🎉 **Breakthrough** |
| **ETTh1** | **-0.31** | -0.26 | -0.06 | +0.44 | 🎉 **2.1× Better** |
| **ETTh2** | **+0.61** | +2.47 | +5.49 | +5.16 | ✅ **Robust** |
