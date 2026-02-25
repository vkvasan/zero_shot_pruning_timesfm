# Technical Specification: Pruning-Time Competitive MoE (branch-moe-interactions)

This document updates the algorithm theory for the current branch. The method is a **pruning-time Mixture-of-Experts (MoE) controller** that selects among multiple pruning experts (`Magnitude`, `Wanda`, `OBS`, `SNR`-biased variants) **per layer**, followed by a **forecast-aware post-pass**.

Unlike standard inference-time MoE, this router is used only during pruning. Deployment remains a single sparse model.

---

## 1. Problem Formulation

Let a dense TimesFM model contain prunable linear layers \(\ell \in \mathcal{L}\). For each layer, the algorithm builds a candidate set \(\mathcal{A}_\ell\) of 2:4 pruning actions:

\[
\mathcal{A}_\ell = \{(\text{expert}, \text{variant})\}
\]

where:
- `expert` ∈ {`Magnitude`, `Wanda`, `OBS`, `SNR`}
- `variant` ∈ {`mask`, `refit`}

The pruning controller chooses one action per layer:

\[
\pi = \{a_\ell\}_{\ell \in \mathcal{L}}
\]

to minimize downstream forecast error (MSE) under a sparsity constraint.

---

## 2. Candidate Generation and Local Reconstruction Objective

For each layer, multiple pruning masks/reconstructions are generated using expert-specific scores. For a candidate \(a \in \mathcal{A}_\ell\), the local proxy objective is a reconstruction loss on cached activations:

\[
\mathcal{L}^{\text{local}}_\ell(a) = \|Y_\ell - \hat{Y}_\ell(a)\|_2^2
\]

where \(Y_\ell\) is the dense layer output on calibration activations and \(\hat{Y}_\ell(a)\) is the candidate pruned/reconstructed output.

This proxy is useful but imperfect: low local reconstruction error does not always imply low forecast MSE after all layers are pruned.

---

## 3. Distribution-Aware Gating Features

The controller computes activation and Gram statistics per layer during calibration:

### 3.1 Activation/Time-Series Features
- **NSR (noise-to-signal ratio)** from energy-band decomposition
- **Trend / season / noise fractions**
- **Activation kurtosis** (heavy-tail behavior)

### 3.2 Gram/Conditioning Features
- **Condition number** (e.g., high percentiles of local Gram conditioning)
- **Diagonal coefficient of variation (CV)**
- **Off-diagonal / diagonal energy ratio**

These features form \(\phi_\ell\), a per-layer summary used to bias expert selection and identify risky layers.

---

## 4. Pruning-Time MoE Routing (Layer-Local Selection)

The first-stage selection chooses a candidate by combining the local proxy with feature-dependent biases/penalties:

\[
a_\ell^{(0)} = \arg\min_{a \in \mathcal{A}_\ell}
\Big(\mathcal{L}^{\text{local}}_\ell(a) + b(a; \phi_\ell)\Big)
\]

Where \(b(a; \phi_\ell)\) is a layer-local prior/penalty that can:
- downweight unstable refits on ill-conditioned layers
- favor robust experts in noisy regimes
- avoid brittle dataset-global hard locks

This branch replaces earlier global heuristics with **layer-local** gating logic.

---

## 5. Risk-Aware Safety Policy (Layer-Level Reversion)

Empirically, a small subset of layers contributes disproportionate forecast error. The safety policy detects these risky layers using \(\phi_\ell\) and reverts them to safer expert/variant choices when locally competitive.

Examples of high-risk patterns:
- noisy early `attn.qkv` / `ff0` layers
- ill-conditioned `ff1` layers
- output projection layers with unstable refit behavior

This acts as a low-cost guardrail before running the post-pass.

---

## 6. Forecast-Aware Greedy Post-Pass (Task Objective Correction)

After layerwise routing, the branch runs a budgeted greedy search on a held-out subset of evaluation windows.

Let \(S\) be the currently selected override set and let \(MSE(S)\) be the forecast MSE with those overrides applied. For a candidate move \(i\):

\[
\Delta_i = MSE(S) - MSE(S \cup \{i\})
\]

Algorithm:
1. Build a risky candidate pool (top-K layers / overrides)
2. Screen by one-layer forecast gain on a small eval subset
3. Apply the best positive-gain move
4. Repeat for a fixed number of steps

This directly targets forecast error and corrects mistakes from local reconstruction proxies.

---

## 7. Pairwise Interaction Diagnostics (Non-Additivity)

Layer decisions interact through residual pathways and downstream activations. Therefore:

\[
\Delta_{i,j} \neq \Delta_i + \Delta_j
\]

The branch includes pairwise diagnostics for screened post-pass moves:

### 7.1 Proxy Interaction (Prediction-Delta Geometry)
For move \(i\), let \(\delta_i\) be the change in model predictions on the eval subset. A proxy synergy score is derived from overlap:

\[
s^{proxy}_{ij} \propto -\langle \delta_i, \delta_j \rangle
\]

- negative dot product → potential synergy (error cancellation)
- positive dot product → potential conflict / overlap

### 7.2 Exact Pair Evaluation (Optional)
For a small subset of move pairs, exact pair MSE is computed:

\[
s^{exact}_{ij} = \big(MSE(\emptyset)-MSE(\{i,j\})\big)-\Delta_i-\Delta_j
\]

This is used for diagnostics and pair-aware greedy ranking.

---

## 8. Practical Implications

### Why this works better than a single pruning method
- Different layer types prefer different experts (`qkv`, `attn.out`, `ff0`, `ff1`, output heads)
- The best choice depends on activation distribution and conditioning, not just weights
- Forecast-aware post-pass repairs the local-vs-global objective mismatch

### What remains hard
- Some `ETTm2` regimes still benefit from stronger post-pass budgets
- Pairwise proxy signs can disagree with exact pair synergy on some moves
- Candidate quality still limits gains when a `SparseGPT`-style solution is not in the candidate set

---

## 9. Deployment and Complexity

- **Training-time / pruning-time cost increases** due to multi-expert evaluation and post-pass search
- **Inference-time cost does not increase** (single pruned model, no router)
- Post-pass cost is controllable via:
  - eval subset size
  - screened pool size
  - greedy step budget
  - candidate cap per layer

---

## 10. Result Reporting (Current Branch Convention)

This branch distinguishes:
- **all-dataset fast sweep** (`results/unified_postpass_all_fast.csv`)
- **targeted strong reruns** (especially for hard `ETTm2` configs)
- **best-available merged result** (`results/sweep_postpass_best_available.csv`)

For branch-level plots and README summaries, use the **best-available merged result**.
