# Tutorial 20: Posterior Downside Probability & Directional Risk

In previous tutorials, we treated predictive epistemic uncertainty symmetrically through standard deviation $\sigma_{\text{epi}}$. However, investors are not averse to upside dispersion; they are averse to **downside loss**.

This tutorial investigates **Posterior Downside Probability**:
> *Does transforming epistemic uncertainty into directional downside probability improve decision relevance and portfolio performance?*

---

## 1. Motivation: Breaking the Symmetry

Consider two investment assets:
- **Asset A**: Large positive forecast $\hat{\mu}_A = +8\%$, with epistemic uncertainty $\sigma_{\text{epi}, A} = 4\%$.
- **Asset B**: Small positive forecast $\hat{\mu}_B = +1\%$, with the exact same uncertainty $\sigma_{\text{epi}, B} = 4\%$.

Under symmetric confidence shrinkage ($\tilde{\mu} = \hat{\mu} / (1 + \lambda \sigma)$) or covariance inflation ($Q + \lambda \sigma^2$), both assets receive the **exact same penalty** ($\sigma_{\text{epi}} = 4\%$).

Yet from an economic perspective:
- Asset A is $2.0$ standard deviations above zero: the probability of a negative outcome is small ($\Phi(-2.0) \approx 2.3\%$).
- Asset B is only $0.25$ standard deviations above zero: the probability of a negative outcome is high ($\Phi(-0.25) \approx 40.1\%$).

Treating uncertainty symmetrically treats Asset A and Asset B identically, ignoring that Asset A has substantial safety margin.

---

## 2. Mathematical Definition

Let $r_{\text{hurdle}}$ be the minimum acceptable return (default $r_{\text{hurdle}} = 0$). The posterior downside probability is:
$$p_{\text{down}, j} = P(\mu_j < r_{\text{hurdle}} \mid \mathcal{D})$$

Under the Gaussian predictive distribution $\mu_j \mid \mathcal{D} \sim \mathcal{N}(\hat{\mu}_j, \sigma_{\text{epi}, j}^2)$:
$$p_{\text{down}, j} = \Phi\left(\frac{r_{\text{hurdle}} - \hat{\mu}_j}{\sigma_{\text{epi}, j}}\right)$$

### Boundary Conditions
- If $\sigma_{\text{epi}, j} \to 0$ and $\hat{\mu}_j > r_{\text{hurdle}}$: $p_{\text{down}, j} \to 0.0$ (certain gain).
- If $\sigma_{\text{epi}, j} \to 0$ and $\hat{\mu}_j < r_{\text{hurdle}}$: $p_{\text{down}, j} \to 1.0$ (certain loss).
- If $\hat{\mu}_j = r_{\text{hurdle}}$: $p_{\text{down}, j} = 0.5$ (neutral coin-flip).

---

## 3. Allocation Mechanisms

We tested two distinct ways to map $p_{\text{down}}$ into portfolio allocation:

### Method 3: Downside-Adjusted Alpha
Multiply expected return by a monotonic dampening factor $g(p_{\text{down}})$:
$$\tilde{\mu}_j = \hat{\mu}_j \cdot \max(0, 1 - \lambda_{\text{down}} \cdot p_{\text{down}, j})$$

When $p_{\text{down}, j} \approx 0$, $g(p) \approx 1$ (zero penalty on confident positive alpha). When $p_{\text{down}, j} \ge 1/\lambda_{\text{down}}$, the asset is shrunk to zero.

### Method 4: Downside Covariance Inflation
Instead of modifying expected return, penalize the quadratic risk matrix along downside-prone assets:
$$Q_{\text{down}} = Q_{\text{market}} + \lambda_{\text{down}} \operatorname{diag}(p_{\text{down}} \cdot \sigma_{\text{epi}}^2)$$

---

## 4. Empirical Downside Calibration (Holdout 2024–2025)

In Phase 3 (Tutorial 16), we established that epistemic uncertainty is highly calibrated against **forecast error magnitude** ($\text{AUROC} = 0.818$).

Now we ask: *Does epistemic uncertainty or $p_{\text{down}}$ predict realized downside?*

| Predictor | Target Event | AUROC | Average Precision | Base Rate |
| :--- | :--- | :---: | :---: | :---: |
| $\sigma_{\text{epi}}$ | Realized Negative Return ($r < 0$) | $0.561 \pm 0.005$ | $0.548$ | $0.504$ |
| $p_{\text{down}}$ | Realized Negative Return ($r < 0$) | $0.562 \pm 0.009$ | $0.549$ | $0.504$ |
| $\sigma_{\text{epi}}$ | Large Downside Move (`label == 1`) | $0.591 \pm 0.004$ | $0.327$ | $0.263$ |
| $p_{\text{down}}$ | Large Downside Move (`label == 1`) | $0.596 \pm 0.002$ | $0.334$ | $0.263$ |

### The Critical Calibration Disconnect
Notice the stark difference:
- **Predicting Error Magnitude**: $\text{AUROC} = \mathbf{0.818}$ (Epistemic uncertainty strongly flags when the neural network is guessing wildly).
- **Predicting Realized Downside**: $\text{AUROC} = \mathbf{0.562 - 0.596}$ (Barely better than random chance at predicting whether the price moves down!).

Epistemic uncertainty measures **model unreliability**, not market direction. A high-uncertainty forecast can result in a massive positive price surge or a catastrophic crash; the dispersion itself does not carry directional signal.

---

## 5. Why Did Downside-Adjusted Alpha (M3) Fail?

On the holdout set, Method 3 produced the worst performance across all policies:
- Return dropped to $27.53\%$ (vs $43.80\%$ in M0).
- Sharpe ratio dropped to $0.974$ (vs $1.519$ in M0).
- Max Drawdown worsened to $-22.75\%$ (vs $-15.12\%$ in M0).

### The High-Alpha / High-Uncertainty Trap
To understand why M3 failed, examine the empirical 4-bin analysis:

| Bin | Mean Realized Return | Mean Weight M0 | Mean Weight M3 |
| :--- | :---: | :---: | :---: |
| **Low Alpha / Low Uncertainty** | $+0.28\%$ to $+0.49\%$ | $3.02\% - 3.86\%$ | $2.34\% - 3.14\%$ |
| **High Alpha / Low Uncertainty** | $\mathbf{+0.70\% \text{ to } +0.88\%}$ | $\mathbf{8.58\% - 9.58\%}$ | $\mathbf{8.45\% - 9.48\%}$ |
| **Low Alpha / High Uncertainty** | $-0.36\%$ to $+0.08\%$ | $1.42\% - 2.37\%$ | $1.13\% - 1.87\%$ |
| **High Alpha / High Uncertainty** | $\mathbf{-1.36\% \text{ to } -1.68\%}$ | $\mathbf{0.85\% - 1.10\%}$ | $\mathbf{2.07\% - 2.82\%}$ |

Look closely at the **High Alpha / High Uncertainty** bin:
1. Realized returns in this bin were deeply negative ($-1.36\%$ to $-1.68\%$).
2. The neural network's point forecast $\hat{\mu}$ was strongly negative ($p_{\text{down}} > 0.5$).
3. Under the deterministic baseline (M0), the optimizer recognized the negative return and allocated a tiny weight ($0.85\% - 1.10\%$).
4. Under Method 3, multiplying $\hat{\mu}$ by $g(p_{\text{down}}) = \max(0, 1 - \lambda p_{\text{down}})$ shrunk the negative return closer to zero!
5. By making a strongly negative return look *less negative* (closer to 0), M3 inadvertently caused the long-only optimizer to allocate **over $2\times$ more capital** to these toxic assets ($2.07\% - 2.82\%$)!

### Conclusion
Attempting to create heuristic directional adjustments from undirected epistemic uncertainty can backfire severely in constrained portfolio optimizers. Shrinking negative returns toward zero reduces the penalty on bad assets, increasing unwanted portfolio exposure.
