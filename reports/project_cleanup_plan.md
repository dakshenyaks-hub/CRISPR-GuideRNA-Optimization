# Project Architecture, Redundancy, Dependency Safety & Cleanup Plan

## Executive Summary

This document establishes the repository-wide architectural audit, dependency mapping, redundancy classification, and safety plan prior to building the production dashboard.

**CRITICAL CLEANUP DIRECTIVE**:  
In accordance with Part 4 of the audit protocol, **no files have been deleted**. This document provides the formal audit trail, evidence, dependency status, and safe deprecation/cleanup roadmap.

---

## 1. Complete File Dependency & Redundancy Classification

All Python modules, scripts, and major assets across the project are categorized into one of five rigorous classifications:
- **A. REQUIRED**: Actively used by the canonical final pipeline.
- **B. HISTORICAL**: Superseded by later steps but preserved for scientific audit, reproducibility, and git history.
- **C. REDUNDANT BUT SAFE TO DEPRECATE**: Obsolete prototype code; can be safely archived to a `legacy/` directory without affecting current pipeline execution.
- **D. REDUNDANT AND SAFE TO REMOVE**: Demonstrably unused, zero historical value, or unversioned archive clutter.
- **E. UNCERTAIN**: Third-party reference submodules or optional assets; retained without modification.

### Complete Inventory Table

| File / Component | Category | Purpose | Imported By / Executed By | Used in Final Pipeline? | Safe Action |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `main.py` | **A** | End-to-end orchestrator for Steps 1–6 (FASTA to features) | CLI entrypoint | **YES** | Keep active |
| `src/gene_sequence_provider.py` | **A** | Dynamic sequence retrieval from local cache & NCBI | `main.py`, `src/multi_gene_validation.py` | **YES** | Keep active |
| `src/ncbi_sequence_provider.py` | **A** | NCBI Entrez E-utilities REST client | `src/gene_sequence_provider.py` | **YES** | Keep active |
| `src/guide_generation.py` | **A** | NGG PAM scanning & candidate extraction | `main.py`, `src/multi_gene_validation.py` | **YES** | Keep active |
| `src/feature_extraction.py` | **A** | SantaLucia $T_m$, ViennaRNA MFE, GC, homopolymers | `main.py`, `src/step9c_cfd.py` | **YES** | Keep active |
| `src/deep_spcas9.py` | **A** | PyTorch DeepSpCas9 CNN activity predictor | `main.py`, `src/multi_gene_validation.py` | **YES** | Keep active |
| `src/cfd_score.py` | **A** | Cutting Frequency Determination (CFD) calculation | `src/step9c_cfd.py`, `tests/test_cfd_score.py` | **YES** | Keep active |
| `src/grna_optimizer.py` | **A** | Multi-objective Pareto front & dominance engine | `scripts/step10b_optimize.py`, `scripts/step11_sensitivity.py` | **YES** | Keep active |
| `scripts/regenerate_step10a_rescored.py` | **A** | Step 10A validated CFD dataset generator | Direct CLI execution | **YES** | Keep active |
| `scripts/step10b_optimize.py` | **A** | Step 10B CFD-aware multi-objective optimizer | Direct CLI execution | **YES** | Keep active |
| `scripts/step11_sensitivity.py` | **A** | Step 11 sensitivity & readiness audit engine | Direct CLI execution | **YES** | Keep active |
| `tests/run_all_tests.py` | **A** | Master test suite runner | Direct CLI execution | **YES** | Keep active |
| `tests/test_cfd_score.py` | **A** | 13 CFD unit test suites | `tests/run_all_tests.py` | **YES** | Keep active |
| `tests/test_cfd_e2e.py` | **A** | 2 CFD end-to-end regression tests | `tests/run_all_tests.py` | **YES** | Keep active |
| `models/.../DeepCas9_weights.pt` | **A** | Converted PyTorch reference neural weights | `src/deep_spcas9.py` | **YES** | Keep active |
| `gene_sequence_provider.py` (root) | **A** | Root convenience re-export | External/interactive | **YES** | Retain |
| `src/multi_gene_validation.py` | **B** | Step 7 multi-gene validation pipeline | Historical CLI runner | **NO** | Keep as Historical |
| `src/step8_benchmarking.py` | **B** | Step 8 multi-gene comparative benchmark | Historical CLI runner | **NO** | Keep as Historical |
| `src/step9_cfd.py` | **B** | Legacy initial CFD script (origin of 999.0 issue) | Historical CLI runner | **NO** | Keep as Historical |
| `src/step9c_cfd.py` | **B** | Intermediate corrected CFD pipeline | `tests/test_cfd_score.py` | **NO** (test only) | Keep as Historical |
| `scripts/regenerate_step9_final.py` | **B** | Intermediate CFD generator (pre-Step 10A) | Historical CLI runner | **NO** | Keep as Historical |
| `src/convert_weights.py` | **B** | One-time TF1 to PyTorch weights converter | One-time execution | **NO** | Keep as Historical |
| `src/inspect_checkpoint.py` | **B** | One-time TF1 checkpoint inspection script | One-time execution | **NO** | Keep as Historical |
| `src/optimization.py` | **C** | Step 4 prototype weighted-sum optimizer | Unused by Steps 7–11 | **NO** | Deprecate to `legacy/` |
| `src/convex_optimization.py` | **C** | Step 5 prototype TP53-only LP ($OT \le 20$) | Unused by Steps 7–11 | **NO** | Deprecate to `legacy/` |
| `src/visualization.py` | **C** | Step 5 prototype plotting for `data/optimized_guides.csv` | Unused | **NO** | Deprecate to `legacy/` |
| `src/convex_visualization.py` | **C** | Step 5 prototype plotting for TP53 LP | Unused | **NO** | Deprecate to `legacy/` |
| `src/create_weighted_vs_convex_selection.py` | **C** | Step 5 prototype comparative figure | Unused | **NO** | Deprecate to `legacy/` |
| `src/build_technical_pdf.py` | **C** | Step 5 prototype PDF technical report builder | Unused | **NO** | Deprecate to `legacy/` |
| `dashboard/create_bio_backend_pdf.py` | **C** | Prototype bio backend PDF builder | Unused | **NO** | Deprecate to `legacy/` |
| `dashboard/index.html` | **C** | Obsolete TP53 prototype HTML dashboard | Hardcoded to Step 5/8 | **NO** | Replace with new App |
| `src/generate_dashboard.py` | **C** | Generator for obsolete `dashboard/index.html` | Unused by final | **NO** | Deprecate to `legacy/` |
| `src/run_dashboard.py` | **C** | Local server for obsolete dashboard (has syntax error) | Broken / Unused | **NO** | Deprecate to `legacy/` |
| `data/gRNA_features_specificity.csv` | **C** | Step 5 2,860-guide TP53 prototype data | Unused | **NO** | Deprecate to `legacy/` |
| `data/optimized_guides.csv` | **C** | Step 5 prototype output | Unused | **NO** | Deprecate to `legacy/` |
| `data/convex_optimized_guides.csv` | **C** | Step 5 prototype output | Unused | **NO** | Deprecate to `legacy/` |
| `src.zip` | **D** | 88 KB redundant zip archive in repo root | None | **NO** | Safe to remove |
| `models/Paired-Library/DeepBaseEditor/` | **E** | Upstream Paired-Library base editor code | External repository | **NO** | Retain untouched |
| `models/Paired-Library/DeepCpf1/` | **E** | Upstream Paired-Library Cas12a/Cpf1 code | External repository | **NO** | Retain untouched |
| `models/Paired-Library/DeepCas9-NG/` | **E** | Upstream Paired-Library Cas9-NG code | External repository | **NO** | Retain untouched |

---

## 2. Evidence-Based Safety Audit

### 2.1 The 4,457 Legacy Issue Resolution
- **Origin**: In `src/step9_cfd.py`, guides that exceeded the Bowtie alignment cap (`-k 1000`) were assigned an arbitrary numerical penalty `cfd_off_target_risk = 999.0`.
- **Exact Count**: In `results/step9_cfd/cfd_guide_features.csv`, exactly 4,457 rows had `cfd_off_target_risk == 999.0`.
- **Supersession**: In Step 10A (`scripts/regenerate_step10a_rescored.py`), artificial penalties were eliminated. Capped guides are explicitly flagged `off_target_capped = True`, `cfd_scoring_complete = False`, and risk is stored as `NaN`.
- **Final Pipeline Isolation**: Verified that `scripts/step10b_optimize.py` and `scripts/step11_sensitivity.py` load **only** `results/step10a_cfd_rescored/cfd_guide_features.csv`, which contains **zero 999.0 values**. The legacy 4,457 issue is completely resolved and poses no risk to the final pipeline.

### 2.2 Dataset Join & Candidate Count Reconciliation
Investigation revealed that the candidate count discrepancy mentioned in previous reports was a textual drafting error in the Step 11 writeup. The physical data files across all stages contain an unvarying, verified number of candidates:

```
results/<GENE>/gRNA_features.csv (Step 7 canonical features)
  ├── TP53  :  3,554 unique guides
  ├── HBB   :    540 unique guides
  ├── EGFR  : 23,112 unique guides
  └── BRCA1 : 15,527 unique guides
       Total: 42,733 unique guides
```

When left-joined with `results/step10a_cfd_rescored/cfd_guide_features.csv` (deduplicated from 59,361 to 42,733 unique `guide_id` keys), there is an exact 1:1 match with **0 missing guides, 0 unmapped keys, and 0 row expansion**.

---

## 3. Recommended Phased Deprecation Plan (DO NOT DELETE NOW)

When project leadership authorizes cleanup, execute the following non-destructive reorganization:

### Phase 1: Archive Redundant Prototypes (Move, Do Not Delete)
Create directory `legacy_archive/` and move:
1. `src/optimization.py` $\to$ `legacy_archive/src/`
2. `src/convex_optimization.py` $\to$ `legacy_archive/src/`
3. `src/visualization.py` $\to$ `legacy_archive/src/`
4. `src/convex_visualization.py` $\to$ `legacy_archive/src/`
5. `src/create_weighted_vs_convex_selection.py` $\to$ `legacy_archive/src/`
6. `src/generate_dashboard.py` $\to$ `legacy_archive/src/`
7. `src/run_dashboard.py` $\to$ `legacy_archive/src/`
8. `data/*.csv` (prototype files) $\to$ `legacy_archive/data/`

### Phase 2: Historical Freezing
Mark the following directories as read-only historical benchmarks:
- `results/step8_benchmarking/`
- `results/step9_cfd/`
- `results/step9_cfd_corrected/`
- `results/step9_cfd_final/`

### Phase 3: Archive Clutter Removal
- Delete `src.zip` from repo root (redundant zip snapshot).

---

## 4. Dashboard Readiness Determination

- **Status**: The existing dashboard (`dashboard/index.html`) is an obsolete Step 5/8 prototype that reflects only TP53, lacks CFD scores, and uses deprecated heuristic filters.
- **Verdict**: A modern, research-grade Multi-Gene Dashboard must be built to interactively showcase the validated Step 10B/11 results.
