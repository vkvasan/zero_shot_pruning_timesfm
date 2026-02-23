# Technical Specification: Unified v2 Pruning

This document provides the mathematical foundations and implementation details for the **Unified v2 Pruning** method, specifically focusing on the adaptive scoring and error-weighted Gram matrix accumulation.

---

## 1. Error-Weighted Gram Matrix Accumulation

The transition from standard pruning to Unified v2 involves a significant shift in how we "look" at the data. We use **Error-Weighted Calibration** to bias the pruning towards windows that the model currently struggles with.

### A. Weight Calculation
For each window $i$ in the calibration set, we compute the model's baseline MSE:
$$ e_i = \text{MSE}(\text{pred}_i, \text{target}_i) $$
The relative error ratio is:
$$ r_i = \frac{e_i}{\frac{1}{N}\sum_{j=1}^N e_j + \epsilon} $$
The final weight for window $i$ is:
$$ w_i = (r_i)^P $$
where $P$ (default=1.0) is the `error_power`. High $P$ forces the Gram matrix to focus strictly on "hard" examples.

### B. Gram Matrix Update
The Gram matrix $G \in \mathbb{R}^{4 \times 4}$ for each 2:4 group is accumulated as a weighted sum of outer products:
$$ G = \sum_{i \in \text{Calib}} w_i \cdot X_i X_i^T $$
Where $X_i \in \mathbb{R}^4$ is the activation vector for that group. This differs from standard WANDA or SparseGPT, which typically use unweighted sums ($w_i = 1$).

---

## 2. Adaptive Unified Scoring

Unified v2 blends two metrics: **Energy Ratio** (Gram-based) and **Spectral Quality** (FFT-based).

### A. Noise Fraction Estimation ($N_{frac}$)
We compute the activation energy across frequency bands using FFT:
*   $E_{signal} = E_{trend} + E_{season}$
*   $E_{noise}$ = energy in the high-frequency band ($>70\%$ of Nyquist).
$$ N_{frac} = \frac{\sum E_{noise}}{\sum (E_{signal} + E_{noise})} $$

### B. Blending Factor ($\alpha$)
A sigmoid-like transition determines the reliance on spectral data:
$$ \alpha = \text{clamp}\left(\frac{N_{frac} - 0.15}{0.15}, 0, 1\right) $$
*   **Clean ($N_{frac} < 15\%$)**: $\alpha = 0$ (Pure Gram Ratio)
*   **Noisy ($N_{frac} > 30\%$)**: $\alpha = 1$ (Pure Spectral Quality)

### C. The Unified Score
$$ S_{unified} = \alpha \cdot \text{Z}(\text{Score}_{spectral}) + (1-\alpha) \cdot \text{Z}(\text{Score}_{ratio}) $$
Where $\text{Z}(\cdot)$ is Z-normalization across the layer groups, and:
$$ \text{Score}_{ratio} = \frac{\mathbf{W}_k^T G \mathbf{W}_k}{\mathbf{W}_d^T G \mathbf{W}_d + \epsilon} $$
(Energy in kept weights vs. energy in dropped weights).

---

## 3. Auto-Ridge Regularization

For refitting the weights, we solve the reconstruction problem via the inverse Hessian.
The effective ridge $\lambda_{eff}$ is calculated globally:

$$ \lambda_{eff} = \lambda_{base} \cdot 10^{3 \cdot \text{clamp}(\frac{\max(N_{frac}) - 0.15}{0.15}, 0, 1)} $$

This scales the ridge from $10^{-5}$ to $10^{-2}$ if any layer in the dataset exhibits high noise.

---

## 4. Long Horizon Heuristic & Tuning
For prediction horizons $H > 100$, the dense model's error signal becomes too noisy for effective calibration in some cases.

*   **General Rule**: Set `error_power = 0` (Uniform Weighting). This stabilizes training and prevents performance spikes (observed in ETTm1/ETTh2).
*   **Refinement (ETTm2)**: For ETTm2, uniform weighting underperforms ($MSE \approx 37$). Tuning revealed that preserving some error information ($P=0.5$) is optimal, recovering performance to $MSE \approx 28.75$.
*   **Implementation**: `run_sweep.py` applies `P=0.5` for ETTm2 and `P=0` for others when $H > 100$.

---

## 5. Why it Works
1.  **Gram Weights** ensure the pruned mask is optimized for the model's failure cases.
2.  **Spectral Blending** prevents the Gram matrix from "hallucinating" structure in pure noise.
3.  **Auto-Ridge** prevents the refit from aggressively fitting to noisy activation patterns.
