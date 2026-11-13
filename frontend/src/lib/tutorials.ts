import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export interface TutorialPart {
  id: string;
  number: number;
  title: string;
  description: string;
}

export const TUTORIAL_PARTS: TutorialPart[] = [
  {
    id: 'part-1',
    number: 1,
    title: 'Foundations & Distributed Formulation',
    description: 'Centralized Markowitz optimization, hypergraph incidence modeling, consensus constraints, primal-dual continuous-time dynamics, and decentralized shadow pricing.',
  },
  {
    id: 'part-2',
    number: 2,
    title: 'Critical Mathematical Audit & Lifted Equivalence',
    description: 'Independent audit of the paper’s reported 0.8573 vs 2.8408 objective value discrepancy, proving informational non-equivalence and deriving the true lifted system.',
  },
  {
    id: 'part-3',
    number: 3,
    title: 'Bayesian Uncertainty & Cross-Sectional Allocation',
    description: 'Ingesting LLM Bayesian-LoRA epistemic uncertainty, testing covariance inflation, downside probability mappings, and synthetic causality controls across 30 seeds.',
  },
  {
    id: 'part-4',
    number: 4,
    title: 'Epistemic Exposure Timing & Falsification',
    description: 'Dynamic exposure scaling, relative abnormality scores, nested walk-forward non-anticipation, 4-week block placebos, lag decay structures, and lead controls.',
  },
  {
    id: 'part-5',
    number: 5,
    title: 'Execution Mechanics & Friction-Aware Timing',
    description: 'Hysteresis state machines, confirmation persistence filters, turnover hurdles, and transaction-cost-aware execution boundaries.',
  },
  {
    id: 'part-6',
    number: 6,
    title: 'Turnover Accounting, Base Churn & Cohort Stacking',
    description: 'Exact turnover accounting identities (risky vs cash), 4-quadrant attribution audits, base portfolio churn diagnostics, and multi-week cohort stacking.',
  },
  {
    id: 'part-7',
    number: 7,
    title: 'Synthesis & The Temporal Regularization Frontier',
    description: 'Phase 20 discovery: overlapping cohort portfolios act as a temporal regularization mechanism, establishing the 5-parameter monotonic optimal horizon frontier.',
  },
];

export interface TutorialMeta {
  slug: string;
  filename: string;
  number: number;
  formattedNumber: string;
  title: string;
  part: TutorialPart;
  description: string;
  readingTimeMinutes: number;
  rawContent: string;
}

export interface TutorialNavigation {
  current: TutorialMeta;
  prev: TutorialMeta | null;
  next: TutorialMeta | null;
}

function resolveTutorialsDir(): string {
  const candidates = [
    path.resolve(process.cwd(), '../tutorials'),
    path.resolve(process.cwd(), 'tutorials'),
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../tutorials'),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(`Tutorials directory not found in candidates: ${candidates.join(', ')}`);
}

function getPartForNumber(num: number): TutorialPart {
  if (num === 9 || num === 13) return TUTORIAL_PARTS[1]; // Part 2: Audit & Equivalence
  if (num === 12 || num === 41) return TUTORIAL_PARTS[6]; // Part 7: Synthesis & Frontier
  if (num <= 8 || num === 10) return TUTORIAL_PARTS[0]; // Part 1: Distributed Foundations
  if (num >= 11 && num <= 21) return TUTORIAL_PARTS[2]; // Part 3: Bayesian Allocation
  if (num >= 22 && num <= 32) return TUTORIAL_PARTS[3]; // Part 4: Timing & Falsification
  if (num >= 33 && num <= 36) return TUTORIAL_PARTS[4]; // Part 5: Execution Mechanics
  if (num >= 37 && num <= 40) return TUTORIAL_PARTS[5]; // Part 6: Accounting & Stacking
  return TUTORIAL_PARTS[0];
}

function cleanTitle(rawFirstLine: string, num: number): string {
  let title = rawFirstLine.replace(/^#\s*/, '').trim();
  // Strip patterns like "Tutorial 00 — ", "Tutorial 41: ", "Tutorial 09 - "
  title = title.replace(/^Tutorial\s+\d+[\s:—–-]+/i, '').trim();
  return title || `Tutorial ${num.toString().padStart(2, '0')}`;
}

function extractDescription(content: string): string {
  const lines = content.split('\n');
  let desc = '';
  let inAlert = false;
  let passedHeader = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('# ')) {
      passedHeader = true;
      continue;
    }
    if (!passedHeader) continue;
    if (line.startsWith('>')) {
      inAlert = true;
      continue;
    }
    if (line === '' || line === '---') {
      inAlert = false;
      continue;
    }
    if (!inAlert && !line.startsWith('#') && !line.startsWith('```') && !line.startsWith('|')) {
      desc = line;
      break;
    }
  }

  // Remove markdown bold/italics/math from description snippet
  desc = desc.replace(/[*_`$]/g, '').trim();
  if (desc.length > 170) {
    desc = desc.slice(0, 167) + '...';
  }
  return desc || 'A mathematically rigorous walkthrough in the Hypergraph Portfolio Lab.';
}

export function getAllTutorials(): TutorialMeta[] {
  const dir = resolveTutorialsDir();
  const filenames = fs.readdirSync(dir).filter((f) => f.endsWith('.md')).sort();

  const tutorials: TutorialMeta[] = filenames.map((filename) => {
    const filePath = path.join(dir, filename);
    const content = fs.readFileSync(filePath, 'utf-8');
    const slug = filename.replace(/\.md$/, '');
    
    const numMatch = filename.match(/^(\d+)_/);
    const num = numMatch ? parseInt(numMatch[1], 10) : 0;
    const formattedNumber = num.toString().padStart(2, '0');

    const firstLine = content.split('\n').find((l) => l.startsWith('# ')) || '';
    const title = cleanTitle(firstLine, num);
    const part = getPartForNumber(num);
    const description = extractDescription(content);

    // Approximate word count
    const words = content.trim().split(/\s+/).length;
    const readingTimeMinutes = Math.max(2, Math.ceil(words / 220));

    return {
      slug,
      filename,
      number: num,
      formattedNumber,
      title,
      part,
      description,
      readingTimeMinutes,
      rawContent: content,
    };
  });

  // Sort numerically
  tutorials.sort((a, b) => a.number - b.number);
  return tutorials;
}

export function getTutorialBySlug(slug: string): TutorialNavigation | null {
  const all = getAllTutorials();
  const index = all.findIndex((t) => t.slug === slug);
  if (index === -1) return null;

  return {
    current: all[index],
    prev: index > 0 ? all[index - 1] : null,
    next: index < all.length - 1 ? all[index + 1] : null,
  };
}

export function getTutorialsByPart(): { part: TutorialPart; tutorials: TutorialMeta[] }[] {
  const all = getAllTutorials();
  return TUTORIAL_PARTS.map((part) => ({
    part,
    tutorials: all.filter((t) => t.part.id === part.id),
  }));
}
