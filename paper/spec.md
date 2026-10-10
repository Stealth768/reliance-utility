# Formal Specification: Reliance-Utility Misalignment in Multimodal AI

## 1. Mathematical Setup and Notation

Let a multimodal input instance be denoted as:
$$X = (X_m, X_{rest})$$
where:
- $X_m$ is the target modality under study (e.g., visual input in Vision-Language Models, camera/LiDAR in autonomous driving).
- $X_{rest}$ denotes all remaining modalities (e.g., textual query/prompt, instructions, metadata).

We define a parameterized corruption operator:
$$T_m(X_m, S)$$
where $S \in [0, 1]$ represents the degradation severity, such that:
- $S = 0$ corresponds to uncorrupted data: $T_m(X_m, 0) = X_m$.
- $S = 1$ corresponds to complete informational degradation of the target modality.

Let $f(\cdot)$ denote a multimodal model producing predictive distribution $B$. The model behavior under corruption severity $S$ is:
$$B_m(S) = f(T_m(X_m, S), X_{rest})$$

We define the null baseline behavior as:
$$B_{null} = f(null_m, X_{rest})$$
where $null_m$ represents the complete absence or neutralization of modality $m$.

---

## 2. Null Modality Variants

To establish robust unimodal baselines $B_{null}$ across different architectures, we evaluate four standard null variants:
1. **Black Image**: Replaces $X_m$ with a zero-valued tensor (all pixels set to 0).
2. **Mean Image**: Replaces $X_m$ with a constant tensor populated by dataset-level channel mean values.
3. **Noise Image**: Replaces $X_m$ with uniform or Gaussian independent identically distributed (i.i.d.) random noise uncorrelated with the task.
4. **Image Tokens Removed**: Completely removes visual tokens/embeddings from the model's sequence representation, evaluating the language model backbone strictly unimodally on $X_{rest}$.

---

## 3. Core Metric Formulations

### 3.1. Modality Utility $U(S)$
Utility measures the task performance benefit gained by providing the degraded modality $X_m$ at severity $S$ relative to the null baseline:
$$U(S) = \text{perf}(B_m(S)) - \text{perf}(B_{null})$$
where $\text{perf}(\cdot)$ evaluates task accuracy (or negative cross-entropy / F1 score) on the target benchmark.
- $U(S) > 0$: The degraded modality still provides net positive task value.
- $U(S) \le 0$: The degraded modality is useless or actively harmful compared to ignoring it altogether.

### 3.2. Information-Theoretic Reliance $R(S)$
Reliance measures the extent to which the model's output distribution shifts when conditioned on the corrupted modality versus the null modality:
$$R(S) = \text{JSD}(B_m(S) \parallel B_{null})$$
where $\text{JSD}(\cdot \parallel \cdot)$ is the Jensen-Shannon Divergence computed in log base 2:
$$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{KL}\left(P \parallel \frac{P + Q}{2}\right) + \frac{1}{2} D_{KL}\left(Q \parallel \frac{P + Q}{2}\right)$$
bounded strictly in $[0, 1]$.

### 3.3. Normalized Reliance $\hat{R}(S)$
To compare models with varying baseline divergence magnitudes, reliance is normalized by its uncorrupted value:
$$\hat{R}(S) = \frac{R(S)}{R(0)}$$
where $R(0) > 0$. When $\hat{R}(S) \approx 1$, the model maintains full behavioral reliance on modality $m$. When $\hat{R}(S) \to 0$, the model effectively discounts modality $m$.

### 3.4. Utility Crossing Point $S^*$
The utility crossing point $S^*$ marks the severity threshold where the modality ceases to deliver positive utility:
$$S^* = \min \{ S \in [0, 1] \mid U_{smooth}(S) \le 0 \}$$
where $U_{smooth}(S)$ is a moving-average smoothed utility curve over window size $w=3$. If $U_{smooth}(S) > 0$ for all $S \in [0, 1]$, then $S^* = \text{None}$.

### 3.5. Normalized Misalignment Index $\text{MI}$
The Misalignment Index quantifies the cumulative unwarranted reliance exhibited by the model over the entire negative-utility regime $[S^*, 1]$:
$$\text{MI} = \frac{1}{1 - S^*} \int_{S^*}^1 \hat{R}(S) \, dS$$
computed numerically via the trapezoid rule.
- If $S^* = \text{None}$, $\text{MI} = \text{None}$.
- If the model dynamically ceases reliance at $S^*$ ($\hat{R}(S) \to 0$ for $S \ge S^*$), then $\text{MI} \approx 0$ (aligned).
- If the model blindly maintains reliance ($\hat{R}(S) \approx 1$ for $S \ge S^*$), then $\text{MI} \approx 1$ (severely misaligned).

### 3.6. Abstention Rate $A(S)$
For models equipped with explicit refusal, abstention, or selective-prediction heads, the abstention rate measures the proportion of samples where the model declines to answer or abstains under severity $S$:
$$A(S) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\text{abstain}(f(T_m(X_{m,i}, S), X_{rest,i}))]$$

---

## 4. Synthetic Benchmark Success Criterion

For the controlled synthetic toy-fusion validation test, empirical success is formalized as:
$$\text{MI}(\text{gated}) < 0.2 \quad \text{and} \quad \text{MI}(\text{fixed}) > 0.7$$
with non-overlapping 95% bootstrap confidence intervals between the dynamically gated fusion architecture and the fixed-weight fusion architecture.
