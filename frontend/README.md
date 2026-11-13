# Hypergraph Portfolio Lab — Research Frontend & Textbook

An Apple-inspired static research website and interactive textbook for the Hypergraph Portfolio Lab, built with [Astro](https://astro.build/) and TypeScript.

This frontend turns the repository's 42 first-principles tutorials and 20+ experimental phases into a navigable, accessible textbook without modifying any underlying research code or duplicating research content into a second source of truth.

---

## Design Principles

- **Apple-Inspired Aesthetic**: Restrained, light-first canvas (`#fbfbfd`), generous whitespace, subtle borders (`rgba(0,0,0,0.08)`), system font stack (`-apple-system`), and responsive typography.
- **Scientific Integrity**: Clear separation between synthetic experiments (Phase 20 DGP), retrospective backtests, and prospective evaluations. Prominent reporting of forensic audits (Phase 16.1 leakage discovery) and negative results (Collaborative LoRA pilot closure).
- **Single Source of Truth**: Reads canonical Markdown files directly from `tutorials/`, `docs/`, and `artifacts/` at build time. No duplicated markdown files or split maintenance.
- **Pure Static Architecture**: 100% pre-rendered HTML and CSS with KaTeX formulas generated at build time. No backend, no API server, no database, no authentication, and no cloud infrastructure requirements.
- **Offline & Self-Contained**: KaTeX fonts and styles are bundled locally. Zero external runtime dependencies or CDNs required.

---

## Directory Structure

```
frontend/
├── astro.config.mjs          # Astro static configuration
├── package.json              # Minimal dependencies & scripts
├── package-lock.json         # Committed npm lockfile
├── tsconfig.json             # Strict TypeScript configuration
├── public/
│   └── favicon.svg           # Hypergraph network node icon
└── src/
    ├── components/
    │   ├── Header.astro      # Translucent navigation header with mobile drawer
    │   ├── Footer.astro      # Site footer with scientific disclaimer & attribution
    │   ├── StatusBadge.astro # Color-coded status pills (Confirmed, Negative, Audited, Mixed)
    │   ├── MethodBadge.astro # Methodology pills (Synthetic, Empirical, Mathematical, Protocol)
    │   ├── TutorialSidebar.astro # 42-chapter navigation sidebar grouped by 7 Parts
    │   └── TableOfContents.astro # Sticky in-page heading navigation with smooth scroll
    ├── layouts/
    │   ├── BaseLayout.astro      # Root layout with metadata and styles
    │   └── TutorialLayout.astro  # 3-column textbook reader layout
    ├── lib/
    │   ├── markdown.ts       # Unified / Remark / Rehype / KaTeX / Callouts processor
    │   ├── tutorials.ts      # Tutorial metadata loader and part grouper
    │   ├── research.ts       # Scientific findings & core questions schema
    │   └── experiments.ts    # 20+ phase registry and artifact doc loader
    ├── pages/
    │   ├── 404.astro         # Clean 404 page
    │   ├── index.astro       # Home landing page & 12 pillars
    │   ├── learn/
    │   │   ├── index.astro   # 42-chapter curriculum index with client-side search
    │   │   └── [slug].astro  # Dynamic chapter reader with LaTeX rendering
    │   ├── research/
    │   │   └── index.astro   # Accessible research syntheses & status matrix
    │   └── experiments/
    │       ├── index.astro   # Filterable experimental phase directory
    │       └── [...path].astro # Artifact & doc report viewer
    └── styles/
        └── global.css        # Apple design tokens, callouts, tables, KaTeX styling
```

---

## Getting Started

### Prerequisites

- Node.js `v20.0+` (tested on Node.js `v26.x`)
- npm `v10.0+`

### Installation

From the repository root or inside `frontend/`:

```bash
cd frontend
npm ci
```

### Development Server

Start a local development server with hot-module reloading:

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

### Type Check & Diagnostics

Run Astro and TypeScript checks:

```bash
npm run check
```

### Production Build

Build the static site into `frontend/dist/`:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

---

## Content Ingestion & Markdown Pipeline

The site uses a custom `unified` pipeline (`src/lib/markdown.ts`) to parse and transform repository Markdown at build time:

1. **`remark-math` + `rehype-katex`**: Math expressions like `$x^T Q x$` and `$$H^* = f(\dots)$$` are converted to standard HTML and MathML.
2. **GitHub Alert Transform**: Transforms blockquotes matching `> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, and `> [!CAUTION]` into styled Apple callout containers.
3. **Relative Link Rewriting**:
   - `01_why_distributed.md` &rarr; `/learn/01_why_distributed`
   - `../docs/collaborative_lora_branch_closure.md` &rarr; `/experiments/docs/collaborative_lora_branch_closure`
   - `../artifacts/.../report.md` &rarr; `/experiments/artifacts/.../report`
4. **Table of Contents Generation**: Extracts all `h2` and `h3` heading elements and builds an interactive in-page table of contents.
