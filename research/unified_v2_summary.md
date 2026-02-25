# Competitive MoE Pruning (TimesFM): Updated Theory Summary

**TL;DR**: This branch treats pruning as a **meta-selection problem**. Instead of committing to one global pruning method (e.g., SparseGPT, Wanda, Magnitude), it selects the **best pruning expert per layer**, then uses a **forecast-aware greedy post-pass** to correct layer decisions that look good locally but hurt end-task forecast accuracy.

---

## 1) What changed from the older “Unified v2/v13” story?

The earlier description emphasized a single unified score (spectral + Gram blending).  
The current branch is more general:

- **Multi-expert candidate set** (`Magnitude`, `Wanda`, `OBS`, `SNR`-biased)
- **Layer-local routing** using activation/Gram statistics
- **Risk-aware safe fallbacks**
- **Forecast-aware multi-layer post-pass**
- **Pairwise interaction diagnostics** (synergy/conflict between layer moves)

This is why the method is better described as a **pruning-time Competitive MoE**.

---

## 2) Core intuition

Different layer types respond differently to pruning:

- `attn.qkv` often prefers robust/noise-aware choices in some regimes
- `attn.out` and `ff0` may prefer simpler masks in others
- `ff1` can be highly ill-conditioned and refit-sensitive
- output projection layers can be numerically unstable

A single pruning method is too rigid. The branch learns to route per layer based on observed activation distributions.

---

## 3) Why local pruning metrics are not enough

Local reconstruction error is useful for ranking candidates, but forecast accuracy depends on **all layers together**.  
Two issues follow:

1. **Proxy mismatch**: the locally best candidate can still increase forecast MSE
2. **Interaction effects**: gains are non-additive across layers

The branch addresses these with:
- a **safety policy** (revert risky layers)
- a **greedy post-pass** (optimize actual forecast MSE)
- **pairwise diagnostics** (measure synergy/conflict)

---

## 4) Why the MoE framing is useful

The MoE framing is not for runtime inference routing. It is a way to formalize:

- a shared **candidate pool of experts**
- a **gate** (layer-local routing)
- a **critic / post-pass** (forecast-level correction)

Result: the system keeps the deployment simplicity of one pruned model while exploiting the strengths of multiple pruning algorithms.

---

## 5) Practical takeaway

The current branch performs best when you think in **three layers of control**:

1. **Expert generation** (MAG / Wanda / OBS / SNR candidates)
2. **Distribution-aware routing** (layer-local gate + safety rules)
3. **Forecast-aware correction** (greedy post-pass, optionally pair-aware)

That is the algorithmic theory to carry forward in docs, plots, and comparisons.
