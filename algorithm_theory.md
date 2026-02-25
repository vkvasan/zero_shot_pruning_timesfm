# Algorithm Theory: Pruning-Time Competitive MoE for TimesFM

This document describes the algorithmic theory used in `branch-moe-interactions`.

## 1. Core Idea

The method is a **pruning-time Mixture-of-Experts (MoE)** controller over pruning algorithms.  
It does **not** add an inference-time MoE router. The final artifact is still a **single pruned sparse model**.

For each prunable layer, the algorithm selects among multiple pruning experts (and variants), then optionally applies a forecast-aware post-pass to improve end-task accuracy.

## 2. Expert Set (Per Layer)

For each layer, candidate 2:4 pruning actions are generated from multiple experts:

- `Magnitude`
- `Wanda`
- `OBS` / Gram-based reconstruction
- `SNR`-biased variants

Each expert may have two variants:

- `mask` (mask only)
- `refit` (mask + local reconstruction/refit)

So the per-layer decision is:

\[
a_\ell \in \{(\text{expert}, \text{variant})\}
\]

## 3. Distribution-Aware Gating (Layer-Local Routing)

The gate uses layer-local activation and Gram statistics gathered on calibration windows, including:

- NSR / energy-band features (trend, season, noise)
- activation kurtosis
- Gram conditioning
- diagonal CV and off-diagonal ratios

These features are used to bias candidate selection and to detect risky layers where local reconstruction proxies are unreliable.

## 4. Local Candidate Selection Objective

The first-stage MoE routing uses a local reconstruction proxy plus feature-based biases/penalties:

\[
a_\ell^{(0)} = \arg\min_{a \in \mathcal{A}_\ell}\; \mathcal{L}^{local}_\ell(a) + b_\ell(a;\phi_\ell)
\]

where:

- \(\mathcal{L}^{local}_\ell(a)\): local reconstruction/validation loss
- \(b_\ell(a;\phi_\ell)\): layer-local bias/penalty using distribution features \(\phi_\ell\)

This replaces brittle dataset-global routing heuristics with **layer-local** decisions.

## 5. Risk-Aware Safety Overrides

Some layers contribute disproportionate forecast error despite good local proxy scores.  
The branch adds a **safety policy** that uses distribution features to detect risky layers and revert them to safer expert/variant choices when locally competitive.

Examples:

- noisy early `attn.qkv` / `ff0` layers
- ill-conditioned `ff1` layers
- unstable output projection layers

## 6. Forecast-Aware Greedy Post-Pass

After layerwise pruning, the method runs a budgeted greedy search on a held-out forecast subset to optimize actual task MSE.

For a candidate override move \(i\), define gain:

\[
\Delta_i = \text{MSE}(S) - \text{MSE}(S \cup \{i\})
\]

where \(S\) is the current set of applied overrides.

Greedy post-pass:

1. Build candidate overrides on risky layers
2. Screen top-K by one-layer forecast gain
3. Apply best positive-gain move
4. Repeat for a small step budget

This corrects local-vs-global mismatch in the first-stage routing.

## 7. Interaction Effects and Pairwise Diagnostics

Layer-level pruning decisions are **non-additive**:

\[
\Delta_{i,j} \neq \Delta_i + \Delta_j
\]

because layer changes interact through residual pathways and downstream activations.

This branch includes pairwise diagnostics for screened post-pass moves:

- **Proxy synergy/conflict** from prediction-delta overlap (dot products / cosine)
- **Optional exact pair evaluations** on a small subset of move pairs

These diagnostics support pair-aware greedy ranking and failure analysis.

## 8. Why This Differs from Prior Single-Method Pruning

Compared to `SparseGPT`, `Wanda`, or `Magnitude` alone, this method:

- treats pruning as a **meta-selection problem**
- selects experts **per layer**, not globally
- uses **distribution-aware routing**
- applies **task-aware post-pass correction**
- explicitly analyzes **pairwise interactions**

## 9. Deployment Property

Despite the MoE framing, deployment remains simple:

- prune once
- save one sparse model
- no runtime router at inference

## 10. Branch Result Convention

This branch distinguishes:

- fast all-dataset sweep (`results/unified_postpass_all_fast.csv`)
- targeted stronger reruns (especially hard `ETTm2` configs)
- merged best-available benchmark (`results/sweep_postpass_best_available.csv`)

Plots and branch-level summaries should use the **best-available merged benchmark**.
