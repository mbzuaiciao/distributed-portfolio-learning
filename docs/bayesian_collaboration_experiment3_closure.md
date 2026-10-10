# Experiment 3 Closure Document: Bayesian Uncertainty Routing

**Repository:** `distributed-portfolio-learning`  
**Branch:** `experiment/bayesian-collaboration-uncertainty-routing`  
**Closure Commit:** `6e2b21c2`  
**Date Closed:** 2026-10-10  
**Protocol Document:** [`docs/bayesian_collaboration_experiment3_plan.md`](bayesian_collaboration_experiment3_plan.md)  
**Amendment Document:** [`docs/bayesian_collaboration_experiment3_amendment.md`](bayesian_collaboration_experiment3_amendment.md)  
**Development Report:** [`artifacts/bayesian_collab_exp3/development_report_exp3.md`](../artifacts/bayesian_collab_exp3/development_report_exp3.md)  
**Status:** **EXPERIMENT 3 CLOSED AS DEVELOPMENT-GATE FAILURE** (Confirmatory test held locked; no parameter changes permitted).

---

## 1. Executive Summary & Closure Decision

Experiment 3 was prospectively designed to test whether Bayesian epistemic uncertainty estimation on specialist representations provides incremental predictive and routing value beyond calibrated confidence, empirical training density, and deterministic mixture-of-experts (MoE) gating.

Under the pre-registered protocol, proceeding to confirmatory held-out testing required passing two mandatory developmental validation gates:
1. **Gate U1 (Support Sparsity Tracking):** $\rho_s(s^2(x), d_M^2(x)) \ge 0.40$
2. **Gate U2 (Residual Error Partial Correlation):** $r(s^2, (p - y)^2 \mid |p - 0.5|) \ge 0.10$

Both developmental gates failed decisively across both experimental regimes (Regime A: axis-aligned regional specialization; Regime B: smooth 2D Gaussian mixture specialization) and across all specialists. In strict adherence to Section 5.3 of the frozen protocol:
- **Execution was halted immediately at Stage 3.**
- **Stage 4 (`evaluate_test`) was NEVER executed.**
- **The quarantined held-out test split ($N=512$, seed=999) remains strictly locked, uninspected, and untouched.**
- **Experiment 3 is formally closed as an unconfirmed development-gate failure.**

---

## 2. Quantitative Gate Outcomes

All metrics evaluated on the disjoint development split ($N=512$, seed=888), averaged over 3 random training seeds ($42, 123, 777$):

| Regime | Specialist | Gate U1 (Support Sparsity $\ge 0.40$) | U1 Verdict | Gate U2 (Residual Partial $\ge 0.10$) | U2 Verdict | Combined Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime A** (Axis-Aligned Shift) | Specialist A ($m_2$) | $\rho_s = 0.250$ | **FAIL** | $r = -0.148$ | **FAIL** | **FAILED** |
| **Regime A** (Axis-Aligned Shift) | Specialist B ($m_3$) | $\rho_s = 0.239$ | **FAIL** | $r = -0.229$ | **FAIL** | **FAILED** |
| **Regime B** (Smooth 2D GMM) | Specialist A ($m_2$) | $\rho_s = 0.170$ | **FAIL** | $r = -0.035$ | **FAIL** | **FAILED** |
| **Regime B** (Smooth 2D GMM) | Specialist B ($m_3$) | $\rho_s = 0.175$ | **FAIL** | $r = -0.156$ | **FAIL** | **FAILED** |

---

## 3. Key Empirical Findings on Development Set

### 3.1 Negligible Value from Epistemic Uncertainty Routing
The prospective design incorporated a matched router ablation specifically to isolate the marginal contribution of epistemic variance inputs:
- $M_{\text{Router-NoUQ}}$: 6 inputs ($l_A, l_B, x_{s,1..4}$), 129 parameters.
- $M_{\text{Router-UQ}}$ ($M_{\text{Bayes-Learn}}$): 8 inputs ($l_A, l_B, s^2_A, s^2_B, x_{s,1..4}$), 161 parameters (+32 linear projection weights).

The empirical difference on the development split was negligible:
- **Regime A:** Brier $M_{\text{Router-NoUQ}}$ (0.174716) vs. $M_{\text{Router-UQ}}$ (0.174634) $\to$ **$\Delta = +0.000082$ Brier**.
- **Regime B:** Brier $M_{\text{Router-NoUQ}}$ (0.187516) vs. $M_{\text{Router-UQ}}$ (0.186421) $\to$ **$\Delta = +0.001094$ Brier**.

Furthermore, the Bayesian precision-weighted log-odds router ($M_{\text{Bayes-Inv}}$) produced substantially higher Brier error in both regimes (0.1991 in Regime A; 0.1957 in Regime B).

### 3.2 Persistent Superiority of Static Calibrated Log-Odds Fusion
Using the pre-registered deterministic comparator selection rule:
- **Regime A:** Selected $M^*$ is `m6_logodds_cal` (**Brier: 0.1646 $\pm$ 0.0062**, AUROC: 0.8218, ECE: 0.0844).
- **Regime B:** Selected $M^*$ is `m6_logodds_cal` (**Brier: 0.1736 $\pm$ 0.0046**, AUROC: 0.7978, ECE: 0.0779).

Static calibrated log-odds fusion (with only 3 parameters: $w_A, w_B, b$) outperformed all learned deterministic routers ($M_{\text{MoE-Lin}}$, $M_{\text{MoE-MLP}}$, $M_{\text{Router-NoUQ}}$), all heuristic routers ($M_{\text{Conf}}$, $M_{\text{Dens}}$), and all Bayesian uncertainty routers ($M_{\text{Bayes-Inv}}$, $M_{\text{Bayes-Learn}}$).

---

## 4. Scientific Scope & Non-Overstatement Safeguards

### 4.1 What This Result Demonstrates
1. Penultimate-layer Laplace approximations with fixed isotropic prior precision ($\alpha = 1.0$) on small ReLU multi-layer perceptrons fail to produce epistemic uncertainty estimates that track input support sparsity ($\rho_s \ll 0.40$).
2. Once predictive confidence $|p - 0.5|$ is controlled for, penultimate Laplace variance correlates *negatively* with residual prediction error ($r \in [-0.23, -0.035]$).
3. Feeding these Laplace variance estimates into a learned router provides no substantive predictive improvement over routing using logits and shared features alone.

### 4.2 What This Result Does NOT Refute
1. **Bayesian Uncertainty in General:** This experiment tests a specific, highly constrained proxy—penultimate Laplace readout on small MLPs with fixed hyperprior. It does *not* refute Bayesian neural networks, full-network Laplace approximations, Hamiltonian Monte Carlo (HMC), variational inference, deep ensembles, or exact Bayesian linear/logistic models.
2. **Specialist Collaboration in General:** Complementary specialist collaboration remains empirically validated by the strong performance of calibrated log-odds fusion ($M_6$, Brier 0.1646 vs. 0.2008 for capacity-matched pooled training). What failed was the specific mechanism of dynamic uncertainty-weighted gating.
3. **Financial Portfolio Utility:** No claims are made regarding financial portfolio optimization, utility, or multi-asset risk allocation.

---

## 5. Frozen Provenance Registry

All Experiment 3 artifacts are permanently versioned at commit `6e2b21c2`:
- `experiments/run_bayesian_collab_experiment3.py` (Stage 2 & Stage 3 code; Stage 4 quarantined)
- `src/hypergraph_portfolio/synthetic_multitask_exp3.py` (Multi-regime synthetic DGP generator)
- `src/hypergraph_portfolio/bayesian_uncertainty.py` (Laplace readouts and router implementations)
- `tests/test_bayesian_uncertainty_exp3.py` (6 unit tests, all passing)
- `artifacts/bayesian_collab_exp3/config_exp3.json`
- `artifacts/bayesian_collab_exp3/dgp_spec_exp3.json`
- `artifacts/bayesian_collab_exp3/frozen_exp3.json`
- `artifacts/bayesian_collab_exp3/development_summary_exp3.json`
- `artifacts/bayesian_collab_exp3/development_metrics_exp3.csv`
- `artifacts/bayesian_collab_exp3/development_evaluation_report_exp3.json`
- `artifacts/bayesian_collab_exp3/development_report_exp3.md`

**Confirmatory held-out test split status:** Strictly locked, untouched, and quarantined.
