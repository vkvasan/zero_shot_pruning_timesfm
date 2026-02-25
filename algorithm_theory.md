# Algorithm Theory: Pruning-Time Competitive MoE (TimesFM)

This document explains the algorithm used in `branch-moe-interactions` in a reader-facing form, with explicit notation and equations.

## 1. What the method is (and is not)

This is a **pruning-time** Mixture-of-Experts (MoE) controller over pruning algorithms.

- It selects the pruning strategy **per layer**.
- It may revise some layer decisions using a small forecast-aware search.
- It produces **one final sparse model**.

It is **not** an inference-time neural MoE (no runtime router is added at inference).

---

## 2. Notation (Plain-English)

This section defines the symbols used later, in plain terms.

- `L` (written as math `𝓛` in formulas): the list of all prunable linear layers in the model.
  - Example: `stacked_xf.7.attn.qkv_proj`, `stacked_xf.7.ff1`, `output_projection_point.output_layer`
- `ℓ` ("ell"): one specific layer chosen from `L`.
- `W_ℓ`: the original dense weight matrix of layer `ℓ` (before pruning).
- `A_ℓ` (math `𝓐_ℓ`): the set of pruning choices we consider for layer `ℓ`.
- `a`: one pruning choice from `A_ℓ`.
  - A pruning choice means **which expert** to use (`Magnitude`, `Wanda`, `OBS`, `SNR`) and **which variant** (`mask` or `refit`).
- `X_ℓ`: calibration inputs going *into* layer `ℓ` (activations at that layer input).
- `Y_ℓ`: outputs of the **dense** version of layer `ℓ` when fed `X_ℓ`.
- `Ŷ_ℓ(a)` (read: "Y-hat for layer ell under action a"): outputs of the **pruned** layer `ℓ` using pruning choice `a`, evaluated on the same `X_ℓ`.

Pruning action format (plain text):

`a = (expert, variant)`

Where:

- `expert ∈ {Magnitude, Wanda, OBS, SNR}`
- `variant ∈ {mask, refit}`

Example:

- `a = (Wanda, refit)` means: prune layer `ℓ` using Wanda-style scoring, then run local refit/reconstruction on the kept weights.

---

## 3. Candidate expert set (per layer)

For every layer \(\ell\), the method constructs multiple 2:4 pruning candidates:

1. **Magnitude**
2. **Wanda**
3. **OBS / Gram-based**
4. **SNR-biased variants**

For many experts, two variants are considered:

- **mask**: apply the 2:4 mask only
- **refit**: apply the mask, then locally reconstruct/refit surviving weights

So the per-layer decision is:

- `a_ℓ`: the pruning action chosen for layer `ℓ`

And the full pruning policy is:

- `π = {a_ℓ for each prunable layer ℓ}`

---

## 4. Local reconstruction objective (first-stage selection)

Each candidate is first evaluated with a **local reconstruction proxy** on cached activations:

\[
\mathcal{L}^{\text{local}}_\ell(a) = \frac{1}{N_\ell}\left\|Y_\ell - \widehat{Y}_\ell(a)\right\|_F^2
\]

where:

- \(N_\ell\) is the number of calibration samples (or normalization factor)
- \(\|\cdot\|_F\) is the Frobenius norm

This proxy is useful, but it is not the final task objective. A candidate with low local reconstruction error can still hurt **forecast MSE** once all layers are pruned.

---

## 5. Distribution-aware gating features (layer statistics)

The method computes layer-local statistics \(\phi_\ell\) from activations and Gram structure.

## 5.1 Gram statistics

Given input activations \(X_\ell\), a Gram / covariance-like matrix is accumulated:

\[
G_\ell = \sum_{i=1}^{N_\ell} w_i \, x_i x_i^\top
\]

where:

- \(x_i\) is the activation vector (or group activation) for sample \(i\)
- \(w_i\) is a calibration weight (uniform when `error_power = 0`)

From \(G_\ell\), the method derives features such as:

- condition number (or robust percentile proxy)
- diagonal coefficient of variation
- off-diagonal / diagonal energy ratio

These indicate numerical stability and refit risk.

## 5.2 Time-series / activation spectral features

For each layer (or representative activations), the method estimates energy in trend/season/noise bands:

\[
E_{\text{total}} = E_{\text{trend}} + E_{\text{season}} + E_{\text{noise}}
\]

and defines a noise-to-signal ratio (NSR):

\[
\text{NSR}_\ell = \frac{E_{\text{noise}}}{E_{\text{trend}} + E_{\text{season}} + \varepsilon}
\]

Additional features include:

- activation kurtosis (heavy tails)
- trend/season/noise fractions

Collect these as:

\[
\phi_\ell = \big[\text{NSR}_\ell,\ \text{kurtosis}_\ell,\ \text{cond}_\ell,\ \text{diagCV}_\ell,\ \text{offdiagRatio}_\ell,\ \dots \big]
\]

---

## 6. Layer-local MoE routing (feature-biased local selection)

The first-stage pruning decision is a feature-biased local selection:

\[
a_\ell^{(0)} = \arg\min_{a \in \mathcal{A}_\ell}
\left(\mathcal{L}^{\text{local}}_\ell(a) + b_\ell(a;\phi_\ell)\right)
\]

where:

- \(\mathcal{L}^{\text{local}}_\ell(a)\) = local reconstruction proxy
- \(b_\ell(a;\phi_\ell)\) = feature-based bias / penalty

Interpretation of \(b_\ell\):

- penalize unstable `refit` variants on ill-conditioned layers
- favor robust experts in noisy regimes
- avoid brittle dataset-global hard locks

This is the **pruning-time MoE router**: it chooses the expert/variant per layer using local evidence + layer statistics.

---

## 7. Risk-aware safety overrides

Some layers have disproportionately high impact on forecast error. The method applies a safety policy on top of the local router.

Define a risk score:

\[
r_\ell = R(\phi_\ell,\ \text{layer\_type}_\ell,\ \text{local margins})
\]

If \(r_\ell\) is high and an alternative is locally competitive, the policy overrides the first-stage choice:

\[
a_\ell^{(1)} =
\begin{cases}
\tilde{a}_\ell, & \text{if } r_\ell > \tau \text{ and } \tilde{a}_\ell \text{ is competitive}\\
a_\ell^{(0)}, & \text{otherwise}
\end{cases}
\]

This is used to prevent known failure modes (e.g., noisy `qkv/ff0`, unstable `ff1`, unstable output projection refits).

---

## 8. Forecast-aware greedy post-pass (task-level correction)

The actual task objective is forecast MSE on held-out windows, not local reconstruction.

Let \(S\) be the current set of post-pass overrides (initially empty after the first-stage policy is applied).  
Let \(\text{MSE}(S)\) denote the forecast MSE with overrides \(S\).

For candidate move \(i\), define its gain:

\[
\Delta_i(S) = \text{MSE}(S) - \text{MSE}(S \cup \{i\})
\]

If \(\Delta_i(S) > 0\), move \(i\) improves forecast accuracy.

### Greedy procedure

1. Build a pool of risky candidate overrides
2. Screen top-\(K\) moves using one-layer forecast gains
3. Repeatedly apply the best positive-gain move
4. Stop at a step budget or when gains fall below threshold

Formally, at greedy step \(t\):

\[
i_t = \arg\max_{i \in \mathcal{P}\setminus S_t} \Delta_i(S_t)
\]

and update:

\[
S_{t+1} = S_t \cup \{i_t\}
\]

if \(\Delta_{i_t}(S_t)\) exceeds a minimum gain threshold.

This stage corrects the mismatch between layer-local proxy quality and end-task forecast MSE.

---

## 9. Interaction effects (non-additivity across layer decisions)

Layer decisions are not additive because changing one layer changes downstream activations and later layer sensitivities.

For two moves \(i\) and \(j\):

\[
\Delta_{i,j}(S) \neq \Delta_i(S) + \Delta_j(S)
\]

where:

\[
\Delta_{i,j}(S) = \text{MSE}(S) - \text{MSE}(S \cup \{i,j\})
\]

Define exact pair synergy:

\[
s^{\text{exact}}_{ij}(S) = \Delta_{i,j}(S) - \Delta_i(S) - \Delta_j(S)
\]

Interpretation:

- \(s^{\text{exact}}_{ij} > 0\): positive synergy (jointly better than additive)
- \(s^{\text{exact}}_{ij} < 0\): conflict / overlap

---

## 10. Proxy pairwise interaction (prediction-delta geometry)

To reduce cost, the branch also estimates pair interactions from prediction deltas on the eval subset.

Let \(p(S)\) be model predictions under override set \(S\).  
Define the prediction delta for move \(i\):

\[
\delta_i = p(S \cup \{i\}) - p(S)
\]

A cheap proxy for synergy uses delta overlap:

\[
s^{\text{proxy}}_{ij} \propto - \langle \delta_i,\ \delta_j \rangle
\]

Intuition:

- if \(\langle \delta_i,\delta_j\rangle < 0\), the moves may cancel errors (synergy)
- if \(\langle \delta_i,\delta_j\rangle > 0\), the moves may overlap/conflict

The branch can use this for diagnostics and pair-aware greedy ranking, with optional exact pair checks on a small screened set.

---

## 11. Why this differs from SparseGPT / Wanda / Magnitude

Single-method pruning uses one pruning rule globally (or per-layer without a task-aware controller).

This method instead solves a **meta-selection** problem:

1. generate multiple expert candidates per layer
2. route per layer using distribution-aware gating
3. apply risk-aware safety overrides
4. correct remaining mistakes with a forecast-aware post-pass
5. analyze pairwise interactions on hard regimes

This is why the approach behaves better across mixed regimes (clean, noisy, ill-conditioned, long-context).

---

## 12. Deployment property and cost

### Inference-time

- one pruned model
- no runtime MoE router
- no per-token expert dispatch

### Pruning-time

Cost increases because the method evaluates multiple experts and runs a post-pass search.  
This cost is controlled by:

- calibration window count
- candidate pool size
- screening top-\(K\)
- post-pass step budget
- eval subset size

---

## 13. Practical result convention for this branch

This branch uses three result tiers:

1. **Baseline full sweep** (`results/restored_v13_sweep.csv`)
2. **Unified post-pass all-dataset fast sweep** (`results/unified_postpass_all_fast.csv`)
3. **Best-available merged benchmark** (`results/sweep_postpass_best_available.csv`)

For reader-facing plots and comparisons, use the **best-available merged benchmark**.
