# Comprehensive Architecture, Redundancy, Dependency Safety & Dashboard Readiness Audit

**Project**: A Multi-Objective Optimization Framework for CRISPR Guide RNA Design Using Machine Learning and Mathematical Optimization  
**Date**: October 2026  
**Status**: Pre-Dashboard Architecture & Readiness Audit Complete  
**Final Scientific Status**: **READY FOR FINAL RESEARCH PRESENTATION**

---

## Executive Summary

Prior to constructing the production dashboard, a repository-wide architectural and dependency audit was executed. This audit rigorously answers the four foundational architectural questions:

1. **What is the TRUE final computational pipeline?**  
   The canonical research pipeline comprises 13 distinct stages orchestrated from target FASTA retrieval (`src/gene_sequence_provider.py`) through biophysical feature extraction (`src/feature_extraction.py`), deep learning cleavage efficiency prediction (`src/deep_spcas9.py`), genome-wide alignment with Bowtie 1.3.1, Cutting Frequency Determination scoring (`src/cfd_score.py` & `scripts/regenerate_step10a_rescored.py`), CFD-aware feasibility filtering and multi-objective Pareto optimization (`scripts/step10b_optimize.py` via `src/grna_optimizer.py`), and multi-scenario sensitivity auditing (`scripts/step11_sensitivity.py`).
2. **Which code/files are redundant or obsolete?**  
   Early development prototypes from Steps 4 and 5 (unconstrained weighted-sum optimization in `src/optimization.py`, heuristic $OT \le 20$ linear programming in `src/convex_optimization.py`, and prototype visualization scripts) are superseded by the Step 10B/11 pipeline.
3. **What can safely be removed, deprecated, or left as historical artifacts?**  
   In strict accordance with audit safety rules, **zero files have been deleted**. Obsolete prototypes are cataloged in `reports/project_cleanup_plan.md` for safe deprecation to an archive folder. Intermediate CFD validation stages (Steps 8, 9, 9C) are preserved as read-only historical milestones for research auditability.
4. **Does a usable dashboard already exist, and if not, exactly what needs to be built?**  
   An early prototype exists (`dashboard/index.html`), but it is **Case B (Partial / Obsolete Prototype)**: it is hardcoded to Step 5 TP53-only data, utilizes deprecated heuristic filters ($OT \le 20$), has zero CFD connectivity, does not support multi-gene operations, and contains server script syntax errors. A production-grade multi-gene dashboard with clean backend decoupling must be constructed.

---

## PART 1 — COMPLETE PROJECT REPOSITORY MAP

```
CRISPR-GuideRNA-Optimization/
├── main.py                                  [Pipeline orchestrator: Steps 1-6]
├── gene_sequence_provider.py                [Root convenience re-export]
├── requirements.txt                         [Core dependencies]
├── walkthrough.md                           [Step-by-step documentation]
├── src/                                     [Core algorithm library]
│   ├── gene_sequence_provider.py            [Dynamic NCBI/local FASTA resolution]
│   ├── ncbi_sequence_provider.py            [NCBI Entrez REST client]
│   ├── guide_generation.py                  [NGG PAM scanning & protospacer extraction]
│   ├── feature_extraction.py                [SantaLucia Tm, ViennaRNA MFE, GC, homopolymers]
│   ├── deep_spcas9.py                       [PyTorch CNN model for DeepSpCas9]
│   ├── cfd_score.py                         [Doench 2016 CFD calculation engine]
│   ├── grna_optimizer.py                    [Multi-objective Pareto dominance engine]
│   ├── multi_gene_validation.py             [Step 7 multi-gene orchestrator - Historical]
│   ├── step8_benchmarking.py                [Step 8 comparative benchmark - Historical]
│   ├── step9_cfd.py                         [Step 9 initial CFD script (999.0 origin) - Historical]
│   ├── step9c_cfd.py                        [Step 9C intermediate corrected CFD - Historical]
│   ├── convert_weights.py                   [One-time TF1 -> PyTorch weight converter]
│   ├── inspect_checkpoint.py                [One-time TF1 checkpoint inspector]
│   ├── optimization.py                      [Step 4 prototype weighted optimizer - Obsolete]
│   ├── convex_optimization.py               [Step 5 prototype LP optimizer - Obsolete]
│   ├── visualization.py                     [Step 5 prototype visualizer - Obsolete]
│   ├── convex_visualization.py              [Step 5 prototype visualizer - Obsolete]
│   ├── create_weighted_vs_convex_selection.py [Step 5 prototype plot - Obsolete]
│   ├── build_technical_pdf.py               [Step 5 prototype PDF builder - Obsolete]
│   ├── generate_dashboard.py                [Step 5/8 prototype dashboard generator - Obsolete]
│   └── run_dashboard.py                     [Step 5/8 prototype local server - Broken]
├── scripts/                                 [Production research pipeline scripts]
│   ├── regenerate_step10a_rescored.py       [Step 10A canonical CFD rescoring]
│   ├── step10b_optimize.py                  [Step 10B canonical multi-objective optimizer]
│   ├── step11_sensitivity.py                [Step 11 sensitivity & readiness audit]
│   └── regenerate_step9_final.py            [Step 9 intermediate generator - Historical]
├── dashboard/                               [Frontend assets]
│   ├── index.html                           [Step 5/8 prototype static HTML - 4.1MB]
│   └── create_bio_backend_pdf.py            [Prototype backend PDF builder]
├── tests/                                   [Test suites]
│   ├── run_all_tests.py                     [Master test runner]
│   ├── test_cfd_score.py                    [13 unit test suites for CFD scoring]
│   └── test_cfd_e2e.py                      [2 end-to-end regression tests]
├── models/Paired-Library/DeepCas9/          [Model artifacts]
│   └── DeepCas9_weights.pt                  [Validated PyTorch CNN weights]
├── results/                                 [Canonical and historical outputs]
│   ├── TP53/, HBB/, EGFR/, BRCA1/           [Step 7 canonical feature tables: gRNA_features.csv]
│   ├── step10a_cfd_rescored/                [Validated CFD dataset: cfd_guide_features.csv]
│   ├── step10b_optimization/                [Validated Pareto fronts & guide selections]
│   ├── step11_sensitivity/                  [Sensitivity scenarios & resolution checks]
│   ├── step8_benchmarking/                  [Historical benchmark outputs]
│   ├── step9_cfd/                           [Historical legacy CFD outputs (contains 999.0)]
│   ├── step9_cfd_corrected/                 [Historical intermediate CFD outputs]
│   └── step9_cfd_final/                     [Historical intermediate CFD outputs]
└── reports/                                 [Audit and analysis reports]
    ├── step10b_optimization_audit.md        [Formal Step 10B audit]
    ├── step11_sensitivity_analysis.md       [Formal Step 11 sensitivity audit]
    └── project_cleanup_plan.md              [Pre-dashboard cleanup roadmap]
```

### Complete Module Inventory and Status Table

| File | Purpose | Used in Final Pipeline? | Imported By | Historical Value | Safe to Remove? |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `main.py` | Pipeline CLI orchestrator (Steps 1–6) | **YES** | None (CLI) | Active | No (Required) |
| `src/gene_sequence_provider.py` | NCBI & local FASTA sequence provider | **YES** | `main.py`, root export | Active | No (Required) |
| `src/ncbi_sequence_provider.py` | E-utilities REST client | **YES** | `src/gene_sequence_provider.py` | Active | No (Required) |
| `src/guide_generation.py` | NGG PAM candidate extractor | **YES** | `main.py` | Active | No (Required) |
| `src/feature_extraction.py` | Biophysical feature computation | **YES** | `main.py`, `src/step9c_cfd.py` | Active | No (Required) |
| `src/deep_spcas9.py` | PyTorch DeepSpCas9 CNN predictor | **YES** | `main.py` | Active | No (Required) |
| `src/cfd_score.py` | Doench CFD scoring engine | **YES** | `src/step9c_cfd.py`, `tests/` | Active | No (Required) |
| `src/grna_optimizer.py` | Pareto dominance & ε-constraint | **YES** | `scripts/step10b_optimize.py`, `scripts/step11_sensitivity.py` | Active | No (Required) |
| `scripts/regenerate_step10a_rescored.py` | Step 10A CFD rescoring generator | **YES** | None (CLI) | Active | No (Required) |
| `scripts/step10b_optimize.py` | Step 10B CFD-aware optimizer | **YES** | None (CLI) | Active | No (Required) |
| `scripts/step11_sensitivity.py` | Step 11 sensitivity audit runner | **YES** | None (CLI) | Active | No (Required) |
| `tests/run_all_tests.py` | Unit & E2E test runner | **YES** | None (CLI) | Active | No (Required) |
| `tests/test_cfd_score.py` | 13 CFD unit test suites | **YES** | `tests/run_all_tests.py` | Active | No (Required) |
| `tests/test_cfd_e2e.py` | 2 CFD regression tests | **YES** | `tests/run_all_tests.py` | Active | No (Required) |
| `src/multi_gene_validation.py` | Step 7 multi-gene orchestrator | No | None (CLI) | High (Step 7) | No (Historical) |
| `src/step8_benchmarking.py` | Step 8 multi-gene benchmark | No | None (CLI) | High (Step 8) | No (Historical) |
| `src/step9_cfd.py` | Step 9 CFD script (999.0 origin) | No | None (CLI) | High (Audit trail) | No (Historical) |
| `src/step9c_cfd.py` | Step 9C intermediate CFD script | No | `tests/test_cfd_score.py` | High (Audit trail) | No (Historical) |
| `scripts/regenerate_step9_final.py` | Intermediate CFD generator | No | None (CLI) | Medium (Audit trail) | No (Historical) |
| `src/convert_weights.py` | TF1 $\to$ PyTorch weights converter | No | None (One-time) | Medium (Setup) | No (Historical) |
| `src/inspect_checkpoint.py` | TF1 checkpoint inspector | No | None (One-time) | Medium (Setup) | No (Historical) |
| `src/optimization.py` | Step 4 weighted-sum optimizer | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `src/convex_optimization.py` | Step 5 TP53-only LP optimizer | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `src/visualization.py` | Step 5 prototype plotting | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `src/convex_visualization.py` | Step 5 prototype LP plotting | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `src/create_weighted_vs_convex_selection.py` | Step 5 comparison figure | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `src/build_technical_pdf.py` | Prototype PDF report generator | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `dashboard/create_bio_backend_pdf.py` | Prototype bio PDF generator | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `dashboard/index.html` | Obsolete TP53 HTML dashboard | No | Browser | Low (Prototype) | Yes (Replace with new app) |
| `src/generate_dashboard.py` | Step 5/8 dashboard generator | No | None | Low (Prototype) | Yes (Deprecate to `legacy/`) |
| `src/run_dashboard.py` | Step 5/8 local server | No | None | Broken (Syntax error) | Yes (Deprecate to `legacy/`) |
| `src.zip` (root) | Stale archive zip file (88 KB) | No | None | None | **YES (Safe to delete)** |

---

## PART 2 — CANONICAL FINAL PIPELINE SPECIFICATION

The canonical research pipeline consists of 13 sequential stages:

```
[Target Gene / FASTA]
        │
        ▼ (Stage 1: Sequence Resolution) -> src/gene_sequence_provider.py
[Full Genomic Sequence]
        │
        ▼ (Stage 2: Candidate Generation) -> src/guide_generation.py
[20nt Protospacers + NGG PAM]
        │
        ▼ (Stage 3: Sequence Filtering) -> src/feature_extraction.py
[GC 30-70%, No Poly-T, Homopolymer <= 4]
        │
        ▼ (Stages 4-5: Biophysical & Structural Features) -> src/feature_extraction.py
[SantaLucia Tm, ViennaRNA MFE, Duplex dG]
        │
        ▼ (Stage 6: Deep Learning Activity Prediction) -> src/deep_spcas9.py
[results/<GENE>/gRNA_features.csv]
        │
        ▼ (Stage 7: Off-Target Alignment) -> Bowtie 1.3.1 (GRCh38.p14)
[SAM/BAM Alignment Records]
        │
        ▼ (Stage 8: Cutting Frequency Determination) -> src/cfd_score.py & scripts/regenerate_step10a_rescored.py
[results/step10a_cfd_rescored/cfd_guide_features.csv]
        │
        ▼ (Stage 9: CFD-Aware Feasibility Filtering) -> scripts/step10b_optimize.py
[Feasible Candidates: Complete CFD, Safe from Censoring]
        │
        ▼ (Stage 10: Multi-Objective Optimization) -> scripts/step10b_optimize.py & src/grna_optimizer.py
[Track A: LP Relaxation (CVXOPT) | Track B: Multi-Objective Frontier]
        │
        ▼ (Stage 11: Pareto Dominance Filtering) -> src/grna_optimizer.find_pareto_front()
[Non-Dominated Pareto Frontier: 8-10 guides/gene]
        │
        ▼ (Stage 12: Sensitivity Analysis & Validation) -> scripts/step11_sensitivity.py
[Adversarial Censored Imputation & Epsilon-Grid Verification]
        │
        ▼ (Stage 13: Final Guide Selection & Ranking) -> scripts/step10b_optimize.py
[Top-5 Balanced Multi-Objective CRISPR Guides]
```

### Canonical Module Execution Map

| Stage | Name | Canonical Script / Module | Input | Output Artifact |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Target Sequence | `src/gene_sequence_provider.py` | Gene symbol (e.g., `TP53`) | `data/reference_sequences/*.fasta` |
| **2** | Candidate Generation | `src/guide_generation.py` | Target FASTA | `results/<GENE>/candidate_gRNAs.csv` |
| **3** | Sequence Filtering | `src/feature_extraction.py` | Candidate table | Intermediate candidate pool |
| **4** | Biophysical Features | `src/feature_extraction.py` | Candidates + FASTA | Thermodynamic parameters ($T_m$, $\Delta G$) |
| **5** | Structural Features | `src/feature_extraction.py` | ViennaRNA secondary model | Folding MFE, paired base fraction |
| **6** | DeepSpCas9 Activity | `src/deep_spcas9.py` | 30bp context sequence | `results/<GENE>/gRNA_features.csv` |
| **7** | Off-Target Alignment | Bowtie 1.3.1 (WSL2) | Protospacer sequences | Alignment BAM/SAM hits |
| **8** | CFD Scoring | `src/cfd_score.py` | Bowtie alignments + reference | `results/step10a_cfd_rescored/cfd_guide_features.csv` |
| **9** | Feasibility Filter | `scripts/step10b_optimize.py` | Enriched feature table | `results/step10b_optimization/<GENE>/run1/feasible_candidates.csv` |
| **10** | Multi-Objective Opt | `scripts/step10b_optimize.py` | Feasible candidates | Track A LP duals & slack values |
| **11** | Pareto Dominance | `src/grna_optimizer.py` | Norm. Activity + CFD Spec. | `results/step10b_optimization/<GENE>/run1/pareto_front.csv` |
| **12** | Sensitivity Audit | `scripts/step11_sensitivity.py` | Step 10B baseline + CFD | `results/step11_sensitivity/step11_sensitivity_summary.json` |
| **13** | Balanced Selection | `scripts/step10b_optimize.py` | Non-dominated front | `results/step10b_optimization/<GENE>/run1/selected_guides.csv` |

---

## PART 3 — REDUNDANCY AUDIT SUMMARY

- **Category A (Required)**: 16 core production components actively supporting the pipeline and tests.
- **Category B (Historical)**: 7 milestone scripts and 4 historical result directories preserved for audit trails.
- **Category C (Redundant but Safe to Deprecate)**: 12 prototype files (from early Step 4/5 iterations) that can be moved to `legacy_archive/`.
- **Category D (Redundant and Safe to Remove)**: 1 item (`src.zip` in root, 88 KB stale zip file).
- **Category E (Uncertain / External)**: Third-party reference submodules in `models/Paired-Library/` (left untouched).

---

## PART 4 — CRITICAL SAFETY COMPLIANCE

In strict compliance with Part 4 of the audit directives:
- **Zero files were deleted during this audit.**
- All file relationships, import graphs, and execution dependencies were empirically verified before drafting recommendations.
- A standalone cleanup plan has been generated at [`reports/project_cleanup_plan.md`](file:///c:/Users/daksh/Documents/CRISPR-GuideRNA-Optimization/reports/project_cleanup_plan.md).

---

## PART 5 — VERIFICATION OF THE 4,457 LEGACY ISSUE

1. **Origin**: In `src/step9_cfd.py`, guides whose off-target alignment count reached the Bowtie reporting ceiling (`-k 1000`) were assigned an artificial penalty `cfd_off_target_risk = 999.0`. In `results/step9_cfd/cfd_guide_features.csv`, exactly 4,457 guides had this value.
2. **Current Status**: In Step 10A (`scripts/regenerate_step10a_rescored.py`), artificial penalty injection was abolished. Capped guides have `off_target_capped = True`, `cfd_scoring_complete = False`, and risk is stored as `NaN`.
3. **Database Audit**:
   - `results/step9_cfd/cfd_guide_features.csv`: 4,457 guides with 999.0.
   - `results/step10a_cfd_rescored/cfd_guide_features.csv`: **0 guides with 999.0**.
4. **Pipeline Isolation**: Neither `scripts/step10b_optimize.py` nor `scripts/step11_sensitivity.py` loads legacy Step 9 files. The current production pipeline has zero dependency on the 4,457 legacy artifact.

---

## PART 6 — DATA SOURCE INTEGRITY AUDIT

We verified the exact file paths loaded by each stage of the current pipeline:

| Pipeline Stage | Target File Loaded | Integrity Verification |
| :--- | :--- | :--- |
| **Step 10B Optimizer** | `results/<GENE>/gRNA_features.csv` | Canonical feature tables from Step 7 (not Step 5 prototype data). |
| **Step 10B Optimizer** | `results/step10a_cfd_rescored/cfd_guide_features.csv` | Validated CFD rescored dataset (deduplicated 1:1 on `guide_id`). |
| **Step 11 Sensitivity** | `results/<GENE>/gRNA_features.csv` | Canonical feature tables. |
| **Step 11 Sensitivity** | `results/step10a_cfd_rescored/cfd_guide_features.csv` | Validated CFD rescored dataset. |

**No active module accidentally loads legacy Step 7 raw unweighted counts, Step 9 prototype CFD, or prototype data from `data/`.**

---

## PART 7 — RESOLUTION OF CANDIDATE COUNT INCONSISTENCIES

### Root Cause Analysis
An inconsistency between previous Step 10B reports (TP53=3554, HBB=540, EGFR=23112, BRCA1=15527) and the Step 11 writeup (TP53=3363, HBB=467, EGFR=24082, BRCA1=27272) was investigated.
- **Physical Verification**: We inspected every CSV, JSON, and database file across the repository.
- **Finding**: The numbers (3,554; 540; 23,112; 15,527) are the **true physical row counts** present in:
  - `results/<GENE>/gRNA_features.csv`
  - `results/multi_gene_validation/step7_report.json`
  - `results/step10a_cfd_rescored/cfd_guide_features.csv` (deduplicated)
  - `results/step10b_optimization/step10b_optimization_summary.json`
  - `results/step11_sensitivity/step11_sensitivity_summary.json` (under `feasibility.total`)
- The alternative numbers (3,363; 467; 24,082; 27,272) were determined to be a **drafting text artifact** in the Step 11 report markdown. They do not correspond to any dataset on disk.
- **Action Taken**: The text tables in `reports/step11_sensitivity_analysis.md` and `step11_final_report.md` have been updated to reflect the canonical verified physical data.

### Canonical Final Results Master Table

| Gene | Total Canonical Candidates | Biophysical & Censoring Excluded | Feasible Pool ($x_i \in \mathcal{S}$) | Verified Pareto Front Size | Selected Balanced Guides ($k=5$) | On-Target Activity Range ($E_i$) | CFD Specificity Range ($S_i$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TP53** | **3,554** | 1,906 | **1,648** | **8** | `TP53_150_-`<br/>`TP53_11911_-`<br/>`TP53_559_-`<br/>`TP53_2766_+`<br/>`TP53_3053_-` | [-0.189, 84.823] | [0.3065, 1.0000] |
| **HBB** | **540** | 175 | **365** | **10** | `HBB_1533_+`<br/>`HBB_3541_+`<br/>`HBB_1202_-`<br/>`HBB_1861_-`<br/>`HBB_2265_-` | [0.800, 81.795] | [0.5838, 0.9945] |
| **EGFR** | **23,112** | 7,022 | **16,090** | **8** | `EGFR_100693_+`<br/>`EGFR_90787_-`<br/>`EGFR_74407_+`<br/>`EGFR_27703_-`<br/>`EGFR_167075_+` | [-3.138, 84.016] | [0.2804, 1.0000] |
| **BRCA1** | **15,527** | 9,749 | **5,778** | **10** | `BRCA1_2088_+`<br/>`BRCA1_2323_+`<br/>`BRCA1_27410_-`<br/>`BRCA1_122416_-`<br/>`BRCA1_74210_+` | [-6.800, 82.355] | [0.1796, 0.9998] |
| **Total** | **42,733** | **18,852** | **23,881** | **36** | **20 Top Guides** | — | — |

---

## PART 8 — DASHBOARD AUDIT FINDINGS

### Status: Case B — Partial / Obsolete Prototype Exists
- **Location**: `dashboard/index.html` (4.1 MB static HTML) + `src/generate_dashboard.py` + `src/run_dashboard.py`.
- **Defects & Limitations**:
  1. **Single-Gene Scope**: Hardcoded exclusively to TP53; cannot display HBB, EGFR, or BRCA1.
  2. **Obsolete Backend Data**: Ingests `data/gRNA_features_specificity.csv` (the early 2,860-guide TP53 prototype), completely bypassing the canonical 3,554 TP53 candidate dataset.
  3. **No CFD Connectivity**: Uses the legacy unweighted off-target count heuristic ($OT \le 20$) and lacks all Cutting Frequency Determination metrics.
  4. **Obsolete Objective Scoring**: Uses the legacy 3-way weighted sum ($0.35 E + 0.40 S + 0.25 R$ with Doench 2016 proxy) rather than DeepSpCas9 and CFD Pareto dominance.
  5. **Syntax Error**: `src/run_dashboard.py` line 34 contains an unterminated string literal (`print("\n`), preventing the local server from starting.

**Conclusion**: The existing dashboard cannot be adapted simply by tweaking configs. A modern, multi-gene dashboard connected to the canonical Step 10B/11 outputs must be built.

---

## PART 9 — FINAL DASHBOARD SPECIFICATION & REQUIREMENTS

The production dashboard must deliver a research-grade experience:

1. **Target Selection**:
   - Interactive selector for validated genes: **TP53**, **HBB**, **EGFR**, **BRCA1**.
   - Input tab for custom gene symbol or FASTA sequence execution.
2. **Interactive Visualizations**:
   - **Multi-Objective Pareto Frontier**: Interactive scatter plot (DeepSpCas9 activity vs. CFD specificity) with non-dominated frontier curve, feasible background cloud, and highlighted Top-5 selections.
   - **Candidate Filtering Funnel**: Step-by-step attrition chart (Total $\to$ GC/Tm $\to$ Homopolymer $\to$ Censored $\to$ Feasible $\to$ Pareto $\to$ Selected).
   - **Biophysical Distribution Radar/Histograms**: GC content, secondary structure MFE, and off-target risk distributions.
3. **Guide Candidate Tables**:
   - Filterable, searchable table of top-ranked and Pareto-optimal guides.
   - Per-guide deep dive: 20nt protospacer sequence, PAM, genomic coordinates, strand, DeepSpCas9 score, CFD off-target risk sum, CFD specificity, and specific off-target site count.
4. **Sensitivity & Robustness Explorer**:
   - Comparison view showing primary vs. censored worst-case stability (demonstrating the 100% selection stability verified in Step 11).
   - $\epsilon$-constraint resolution stability toggle (50 vs. 100 vs. 200 steps).
5. **Scientific Disclaimer**:
   > *All guide scores, on-target cleavage predictions, and off-target specificity metrics are computational outputs generated by machine learning models (DeepSpCas9) and biophysical empirical matrices (CFD). They represent mathematical optimization and do not constitute wet-lab experimental validation.*

---

## PART 10 — DASHBOARD BACKEND INTERFACE DESIGN

To ensure strict separation of concerns, the dashboard will **not** duplicate research algorithms. It will interface with a clean controller: `src/pipeline_controller.py`:

```python
class PipelineController:
    @staticmethod
    def get_gene_summary(gene: str) -> dict:
        """Loads canonical Step 10B/11 cached optimization and sensitivity results."""
        ...
        
    @staticmethod
    def get_feasible_candidates(gene: str) -> pd.DataFrame:
        """Returns the feasible candidate dataset with normalized metrics."""
        ...

    @staticmethod
    def get_pareto_front(gene: str) -> pd.DataFrame:
        """Returns the verified non-dominated Pareto frontier."""
        ...

    @staticmethod
    def get_selected_guides(gene: str) -> pd.DataFrame:
        """Returns the top-5 balanced selected guides with complete annotations."""
        ...

    @staticmethod
    def run_custom_pipeline(sequence: str, config: dict) -> dict:
        """Executes candidate extraction, DeepSpCas9 inference, and optimization for custom FASTA."""
        ...
```

---

## PART 11 — FINAL PROPOSED ARCHITECTURE

```text
                             USER INTERFACE
             (Modern Interactive Web Dashboard / SPA)
                                  │
                                  ▼
                         PIPELINE CONTROLLER
                     (src/pipeline_controller.py)
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼ (Cached Mode)                                 ▼ (Live Inference Mode)
CANONICAL DATA STORE                            COMPUTATIONAL ENGINE
├── results/step10b_optimization/               ├── Target Resolution: src/gene_sequence_provider.py
├── results/step11_sensitivity/                 ├── Candidate Extraction: src/guide_generation.py
├── results/step10a_cfd_rescored/               ├── Feature Extraction: src/feature_extraction.py
└── reports/*.json                              ├── DeepSpCas9 CNN: src/deep_spcas9.py
                                                ├── CFD Scoring: src/cfd_score.py
                                                └── Pareto Engine: src/grna_optimizer.py
                                  │
                                  ▼
                         RESULT DATA MODEL
                 (JSON Payload / Interactive State)
                                  │
                                  ▼
                         DASHBOARD PRESENTATION
                 ├── Multi-Gene Benchmark Overview
                 ├── Interactive Pareto Trade-off Explorer
                 ├── Biophysical Filtering Funnel
                 ├── Candidate Sequence Inspector
                 └── Sensitivity & Robustness Audit View
```

**Next Step Recommendation**: Proceed directly to implementing the unified `src/pipeline_controller.py` and constructing the modern research dashboard.
