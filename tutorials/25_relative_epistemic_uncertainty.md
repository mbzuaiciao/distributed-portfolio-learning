# Tutorial 25: Relative Epistemic Uncertainty & Abnormality Detection

In Phase 5, we discovered that absolute epistemic uncertainty ($u_t$) dampened portfolio volatility by $70\%$ (from $29.1\%$ down to $8.7\%$) and reduced maximum drawdown from $-15.1\%$ to $-4.5\%$. However, exposure attribution revealed an underlying limitation:
- **Exposure before worst 10% weeks**: $28.8\%$
- **Exposure before best 10% weeks**: $31.7\%$
- **Average exposure**: $30.1\%$

Because the raw scalar level of $u_t$ remained consistently positive throughout the market cycle, absolute scaling caused **nearly permanent de-risking**. The portfolio held $\approx 70\%$ cash at all times, dampening bull run gains along with bear crashes.

To extract true dynamic timing, we must normalize uncertainty relative to trailing market history.

---

## 1. Why Absolute Uncertainty Induces Structural De-risking

Let $\sigma_{\text{epi}, j, t}$ be the epistemic standard deviation produced by a Bayesian-LoRA Monte Carlo ensemble. The cross-sectional 75th percentile aggregate is:
$$u_t = \operatorname{percentile}_{75}\left(\{\sigma_{\text{epi}, j, t}\}_{j=1}^m\right)$$

In financial time series, neural network models maintain a persistent baseline uncertainty due to parameter uncertainty and latent market noise. Under the absolute rational shrinkage rule:
$$s_{\text{epi}, t} = \frac{1}{1 + \lambda_{\text{epi}} u_t}$$

When tuned on development data, the optimizer selects a high penalty ($\lambda_{\text{epi}} = 300.0$) to satisfy risk aversion, which forces $s_{\text{epi}, t} \approx 0.30$ permanently. The strategy behaves almost identically to a **fixed 30% equity / 70% cash allocation**.

---

## 2. Formulating Causal Trailing Normalizations

To test whether *abnormally elevated* model uncertainty indicates oncoming market distress, we construct relative uncertainty metrics using **strictly trailing history up to $t-1$** (zero lookahead):

### Candidate 1: Rolling Sample Z-Score
$$z_{\text{epi}, t} = \frac{u_t - \bar{u}_{t-1, W}}{\hat{\sigma}_{t-1, W}}$$
Where:
- $\bar{u}_{t-1, W} = \frac{1}{W} \sum_{k=1}^W u_{t-k}$
- $\hat{\sigma}_{t-1, W} = \sqrt{\frac{1}{W-1} \sum_{k=1}^W (u_{t-k} - \bar{u}_{t-1, W})^2}$

### Candidate 2: Robust Rolling Median Absolute Deviation (MAD) Z-Score
Outlier spikes in uncertainty can distort standard deviation. The robust z-score normalizes by the trailing median and scaled MAD:
$$z_{\text{epi, robust}, t} = \frac{u_t - \operatorname{median}_{t-1, W}(u)}{1.4826 \cdot \operatorname{MAD}_{t-1, W}(u)}$$
Where $\operatorname{MAD}_{t-1, W}(u) = \operatorname{median}_{t-1, W}\left(|u - \operatorname{median}_{t-1, W}(u)|\right)$.

### Candidate 3: Ratio to Trailing Baseline
$$u_{\text{rel}, t} = \frac{u_t}{\operatorname{median}_{t-1, W}(u)}$$

### Candidate 4: Trailing Percentile Rank
$$u_{\text{rank}, t} = \frac{1}{W} \sum_{k=1}^W \mathbf{1}_{\{u_t \ge u_{t-k}\}} \in [0, 1]$$

---

## 3. Development Window & Metric Selection (2022–2023)

We evaluated all four normalization methods across lookback windows $W \in [8, 13, 26, 52]$ on the 99-week development dataset.

The **Robust MAD Z-Score with $W = 8$ weeks** achieved the highest discriminative power for detecting oncoming negative return weeks ($\text{AUROC} = 0.625$ vs $0.541$ for absolute uncertainty).

### Causal Invariants & Edge Cases:
1. **Minimum History Requirement**: For $t < 4$, the function returns a neutral default ($z_t = 0.0$) rather than computing an unstable estimate.
2. **Zero-MAD Robustness**: If trailing uncertainty is constant ($\text{MAD} = 0$), the function falls back to sample standard deviation; if variance is also zero, it returns $0.0$, preventing division-by-zero crashes.
3. **Zero Lookahead**: $u_t$ is normalized against $\{u_{t-W}, \dots, u_{t-1}\}$. Future and contemporaneous values are excluded from the normalization baseline.

In Tutorial 26, we show how to map $z_{\text{epi}, t}$ into a dynamic portfolio exposure scale while strictly matching the average exposure budget of control benchmarks.
