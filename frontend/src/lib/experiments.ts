import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export interface ExperimentPhase {
  id: string;
  phaseNumber: string;
  title: string;
  adjudicatedOutcome: string;
  status: 'confirmed' | 'negative' | 'audited' | 'mixed';
  statusLabel: string;
  methodology: 'synthetic' | 'empirical' | 'mathematical' | 'protocol';
  methodologyLabel: string;
  summary: string;
  primaryMetric: string;
  reports: {
    title: string;
    path: string;
    type: 'report' | 'audit' | 'preregistration' | 'derivation' | 'protocol';
  }[];
}

export const EXPERIMENT_PHASES: ExperimentPhase[] = [
  {
    id: 'phase-20',
    phaseNumber: 'Phase 20',
    title: 'Optimal Cohort Holding Horizon & Temporal Regularization Frontier',
    adjudicatedOutcome: 'Outcome A (Confirmed Frontier)',
    status: 'confirmed',
    statusLabel: 'Outcome A — Confirmed',
    methodology: 'synthetic',
    methodologyLabel: 'Synthetic DGP (30 Seeds, 288k Cells)',
    summary: 'Proves that overlapping cohort portfolios act as a temporal regularization mechanism. Confirms all 5 comparative statics (cost, persistence, noise, alpha reversal, risk aversion) with zero accounting error (|error| <= 3.14e-17).',
    primaryMetric: 'Accounting closure error < 1e-16; 5/5 comparative statics confirmed at 90-100% consistency',
    reports: [
      {
        title: 'Scientific Report: Optimal Cohort Holding Horizon',
        path: 'artifacts/phase20_optimal_cohort_horizon/scientific_report',
        type: 'report',
      },
      {
        title: 'Null Regime Invariant Audit',
        path: 'artifacts/phase20_optimal_cohort_horizon/null_regime_invariant_audit',
        type: 'audit',
      },
      {
        title: 'Analytical Derivation & Utility Model',
        path: 'artifacts/phase20_optimal_cohort_horizon/analytical_derivation',
        type: 'derivation',
      },
      {
        title: 'Phase 20 Pre-Registration Document',
        path: 'artifacts/phase20_optimal_cohort_horizon/preregistration',
        type: 'preregistration',
      },
    ],
  },
  {
    id: 'phase-19',
    phaseNumber: 'Phase 19',
    title: 'Architecture & Signal Stress Test',
    adjudicatedOutcome: 'Outcome D (Mixed / Inconclusive)',
    status: 'mixed',
    statusLabel: 'Outcome D — Mixed',
    methodology: 'synthetic',
    methodologyLabel: 'Synthetic Parameter Sweep (512 Evaluations)',
    summary: 'Confirms architecture-general turnover reduction (>=73.97%) and volatility dampening (63.3%-69.7%), but revealed unexpected reversal: H=4 advantage weakens as signal strength increases.',
    primaryMetric: 'Turnover -74% across all regimes; positive delta utility under weak alpha, negative under strong alpha',
    reports: [
      {
        title: 'Scientific Report: Signal Stress Test',
        path: 'artifacts/phase19_architecture_signal_stress_test/scientific_report',
        type: 'report',
      },
      {
        title: 'Pre-registration Protocol',
        path: 'artifacts/phase19_architecture_signal_stress_test/preregistration',
        type: 'preregistration',
      },
    ],
  },
  {
    id: 'phase-18',
    phaseNumber: 'Phase 18',
    title: 'Clean Cohort Revalidation',
    adjudicatedOutcome: 'Outcome B (Architecture Only)',
    status: 'mixed',
    statusLabel: 'Outcome B — Architecture Only',
    methodology: 'empirical',
    methodologyLabel: 'Historical Reconstructed Backtest',
    summary: 'Structural mechanics confirmed: turnover reduced by 73.97%, portfolio volatility reduced by 66.21%, effective holdings tripled. Net economic alpha did not pass statistical significance (CI [-54.68, +84.33] bp).',
    primaryMetric: 'Delta utility +13.80 bp/wk, p = 0.105; turnover -73.97%; volatility -66.21%',
    reports: [
      {
        title: 'Scientific Report: Clean Cohort Revalidation',
        path: 'artifacts/phase18_clean_cohort_revalidation/scientific_report',
        type: 'report',
      },
      {
        title: 'Provenance Audit',
        path: 'artifacts/phase18_clean_cohort_revalidation/provenance_audit',
        type: 'audit',
      },
      {
        title: 'Compute Cost Ledger',
        path: 'artifacts/phase18_clean_cohort_revalidation/compute_cost_report',
        type: 'report',
      },
    ],
  },
  {
    id: 'collaborative-lora',
    phaseNumber: 'Pilot',
    title: 'Collaborative LoRA Specialist Coordination Pilot',
    adjudicatedOutcome: 'Closed — Negative Result',
    status: 'negative',
    statusLabel: 'Closed — Negative Result',
    methodology: 'protocol',
    methodologyLabel: 'Pre-registered Multi-Task LoRA Pilot',
    summary: 'Evaluated whether sector LoRA specialists improve held-out target tasks. Learned weights degenerated to banking-only selection, CoLoRA failed all continuation criteria, and the prevalence baseline remained superior.',
    primaryMetric: 'Mean Brier: Target 0.1937, Pooled 0.1964, CoLoRA 0.1964, Prevalence Null 0.1841',
    reports: [
      {
        title: 'Collaborative LoRA Branch Closure Summary',
        path: 'docs/collaborative_lora_branch_closure',
        type: 'report',
      },
      {
        title: 'Mechanism Pilot Report',
        path: 'artifacts/collaborative_lora_mechanism_pilot/report',
        type: 'report',
      },
      {
        title: 'Pilot Protocol & Stopping Rules',
        path: 'artifacts/collaborative_lora_mechanism_pilot/protocol',
        type: 'protocol',
      },
    ],
  },
  {
    id: 'phase-16-1',
    phaseNumber: 'Phase 16.1',
    title: 'Forensic Audit & Deterministic Control',
    adjudicatedOutcome: 'Outcomes A & C (Audited & Discarded)',
    status: 'audited',
    statusLabel: 'Audited & Corrected',
    methodology: 'empirical',
    methodologyLabel: 'Forensic Training Leakage Audit',
    summary: 'Discovered that training text in Phase 11/16 leaked 1-day returns (move_1d). Proved that clean regularized deterministic LoRA strictly outperforms Bayesian LoRA across all calibration and discrimination metrics.',
    primaryMetric: 'Clean Deterministic vs Bayesian: AUROC 0.5420 vs 0.5001, Brier 0.1597 vs 0.1932, Log Loss 0.4998 vs 1.0804',
    reports: [
      {
        title: 'Phase 16.1 Audit Report',
        path: 'artifacts/phase16_1_audit/phase16_1_report',
        type: 'audit',
      },
      {
        title: 'Deterministic Control Report',
        path: 'artifacts/phase16_deterministic_control/phase16_report',
        type: 'report',
      },
    ],
  },
  {
    id: 'phase-15',
    phaseNumber: 'Phase 15',
    title: 'Incremental Uncertainty & Provenance Audit',
    adjudicatedOutcome: 'Audit Complete',
    status: 'audited',
    statusLabel: 'Audited',
    methodology: 'empirical',
    methodologyLabel: 'Uncertainty Ablation & Data Tracing',
    summary: 'Audited data provenance and incremental uncertainty contributions across asset groups, establishing data boundaries and verifying zero-leakage test protocols.',
    primaryMetric: 'Full provenance trace verified across all training and test event IDs',
    reports: [
      {
        title: 'Phase 15 Report: Incremental Uncertainty',
        path: 'artifacts/phase15_incremental_uncertainty/phase15_report',
        type: 'report',
      },
      {
        title: 'Provenance Audit Report',
        path: 'artifacts/phase15_1_provenance_audit/phase15_1_report',
        type: 'audit',
      },
    ],
  },
  {
    id: 'phase-13',
    phaseNumber: 'Phase 13',
    title: 'Cloud Execution Addendum & Hardware Profiling',
    adjudicatedOutcome: 'Infrastructure Verified',
    status: 'confirmed',
    statusLabel: 'Infrastructure Confirmed',
    methodology: 'empirical',
    methodologyLabel: 'Apple Silicon M2 MPS vs GCE NVIDIA L4',
    summary: 'Profiled local Apple Silicon (M2 MPS) versus cloud spot instances (Google Cloud NVIDIA L4). Established preemption-safe sharding, automated cost tracking, and strict zero-market-data-transfer protocols.',
    primaryMetric: 'Execution parity verified with zero market-data transfer to cloud environments',
    reports: [
      {
        title: 'Cloud Execution Addendum Report',
        path: 'artifacts/phase13_uncertainty_mapping/phase13_cloud_addendum_report',
        type: 'report',
      },
      {
        title: 'Data Scope Report',
        path: 'artifacts/phase13_uncertainty_mapping/phase13a_data_scope_report',
        type: 'audit',
      },
    ],
  },
  {
    id: 'phase-11-12',
    phaseNumber: 'Phase 11–12',
    title: 'Replication & Prospective Evaluation',
    adjudicatedOutcome: 'Replication Passed / Prospective Inconclusive',
    status: 'mixed',
    statusLabel: 'Historical Endpoint Replicated',
    methodology: 'protocol',
    methodologyLabel: 'Strict Prospective Freeze Protocol',
    summary: 'Confirmed numerical reproduction of historical endpoints. Established prospective freeze protocol (2025-05-15 cutoff). Out-of-sample prospective holdout did not yield statistically significant alpha.',
    primaryMetric: 'Machine-precision numerical reproduction of Phase 10 endpoints',
    reports: [
      {
        title: 'Phase 11 Replication Report',
        path: 'artifacts/phase11_replication/phase11_replication_report',
        type: 'report',
      },
      {
        title: 'Phase 12 Prospective Evaluation Report',
        path: 'artifacts/phase12_prospective_evaluation/phase12_prospective_report',
        type: 'report',
      },
    ],
  },
  {
    id: 'phase-10',
    phaseNumber: 'Phase 10',
    title: '2026 Frozen Confirmation Protocol',
    adjudicatedOutcome: 'Audited & Documented',
    status: 'audited',
    statusLabel: 'Frozen Protocol Audited',
    methodology: 'protocol',
    methodologyLabel: 'Pre-registered Manifest & Data Cutoff Audit',
    summary: 'Pre-registered the frozen architecture manifest (H=4, M13 Hysteresis tau_enter=1.2, tau_exit=0.2, 10 bps costs). Data audit confirmed historical returns terminate on 2025-05-15.',
    primaryMetric: 'Pre-registered manifest signed prior to holdout unblinding',
    reports: [
      {
        title: 'Phase 10 Confirmation Report',
        path: 'artifacts/phase10_2026_confirmation/phase10_confirmation_report',
        type: 'report',
      },
      {
        title: 'Model Provenance Report',
        path: 'artifacts/phase10_1_model_provenance/phase10_1_provenance_report',
        type: 'audit',
      },
    ],
  },
];

function resolveRepoRoot(): string {
  const candidates = [
    path.resolve(process.cwd(), '..'),
    process.cwd(),
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..'),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, 'tutorials')) && fs.existsSync(path.join(candidate, 'artifacts'))) {
      return candidate;
    }
  }
  return path.resolve(process.cwd(), '..');
}

export interface ArtifactFile {
  relativePath: string;
  absolutePath: string;
  title: string;
  category: 'artifact' | 'doc';
  content: string;
}

export function getAllArtifactDocs(): ArtifactFile[] {
  const root = resolveRepoRoot();
  const results: ArtifactFile[] = [];

  // Read docs/
  const docsDir = path.join(root, 'docs');
  if (fs.existsSync(docsDir)) {
    const docFiles = fs.readdirSync(docsDir).filter((f) => f.endsWith('.md'));
    for (const f of docFiles) {
      const p = path.join(docsDir, f);
      const content = fs.readFileSync(p, 'utf-8');
      const firstLine = content.split('\n').find((l) => l.startsWith('# ')) || f;
      results.push({
        relativePath: `docs/${f.replace(/\.md$/, '')}`,
        absolutePath: p,
        title: firstLine.replace(/^#\s*/, '').trim(),
        category: 'doc',
        content,
      });
    }
  }

  // Read artifacts/
  const artifactsDir = path.join(root, 'artifacts');
  if (fs.existsSync(artifactsDir)) {
    const subdirs = fs.readdirSync(artifactsDir);
    for (const sub of subdirs) {
      const subPath = path.join(artifactsDir, sub);
      if (fs.statSync(subPath).isDirectory()) {
        const files = fs.readdirSync(subPath).filter((f) => f.endsWith('.md'));
        for (const f of files) {
          const p = path.join(subPath, f);
          const content = fs.readFileSync(p, 'utf-8');
          const firstLine = content.split('\n').find((l) => l.startsWith('# ')) || f;
          results.push({
            relativePath: `artifacts/${sub}/${f.replace(/\.md$/, '')}`,
            absolutePath: p,
            title: firstLine.replace(/^#\s*/, '').trim(),
            category: 'artifact',
            content,
          });
        }
      }
    }
  }

  return results;
}

export function getArtifactByPath(relPath: string): ArtifactFile | null {
  const docs = getAllArtifactDocs();
  const clean = relPath.replace(/\.md$/, '').replace(/^\/+/, '');
  return docs.find((d) => d.relativePath === clean) || null;
}
