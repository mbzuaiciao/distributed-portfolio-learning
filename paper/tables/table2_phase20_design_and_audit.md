# Table 2: Phase 20 Synthetic Factorial Simulation Design & Invariant Audit

| Parameter / Audit Dimension | Specification / Value | Description & Guardrail |
| :--- | :--- | :--- |
| **Asset Universe Size ($N$)** | 30 assets | Fixed synthetic cross-section |
| **Evaluation Timeline ($T$)** | 156 calendar weeks | 3 years of weekly market returns |
| **Latent History Truncation ($K$)** | 52 weeks | Truncation weight at $\rho=0.97$: $0.97^{52} \approx 0.2032$ (finite-memory approximation) |
| **Market Factor Shock ($\sigma_{\text{mkt}}$)** | 0.02 (200 bps/week) | Shared calendar-week market innovation |
| **Idiosyncratic Shock ($\sigma_{\text{idio}}$)** | 0.04 (400 bps/week) | Asset-specific calendar return shock |
| **Simulation Seeds ($S$)** | 30 independent seeds | Seeds 200001 through 200030 inclusive |
| **Candidate Horizons ($H$)** | $\{1, 2, 4, 8, 13\}$ weeks | Frozen candidate set (5 levels) |
| **Signal Magnitude ($\beta$)** | $\{0.000, 0.005, 0.010, 0.020\}$ | 4 levels; deactivates to uniform weighting at $\beta=0.000$ |
| **Persistence ($\rho$)** | $\{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$ | 6 levels; continuous half-life $\tau_{1/2} \in [0.0, 22.8]$ weeks |
| **Observation Noise ($\sigma_{\text{noise}}$)** | $\{0.0, 0.5, 1.0, 2.0\}$ | 4 levels of additive ranking error |
| **Transaction Cost ($c$)** | $\{0, 5, 10, 25, 50\}$ bps | 5 levels of linear proportional friction |
| **Risk Aversion ($\gamma$)** | $\{0.0, 0.5, 1.0, 2.5, 5.0\}$ | 5 levels of mean-variance risk penalty |
| **Total Factorial Evaluations** | **360,000 evaluations** | $4 \times 6 \times 4 \times 5 \times 5 \times 5 \times 30$ full factorial grid |
| **Pairwise Evaluations vs $H=1$** | **288,000 comparisons** | Evaluated across $H \in \{2, 4, 8, 13\}$ |
| **Maximum Accounting Error** | **$3.144 \times 10^{-17}$** | Strictly satisfies tolerance $|\text{error}| < 10^{-12}$ |
| **Exact-Null ($\beta=0$) Discrepancy** | **$0.00\text{e}+00$** | Verified exact across 7 invariants; 100% classified as `Tie_1-2-4-8-13` |
| **Adjudicated Outcome** | **Outcome A** | Preregistered criteria satisfied: accounting closure, verified invariants, confirmed statics |
