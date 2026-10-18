# Dashboard Integration Audit: Pre-Implementation Architecture, Data Sources & Controller Specification

**Project**: A Multi-Objective Optimization Framework for CRISPR Guide RNA Design Using Machine Learning and Mathematical Optimization  
**Document**: Pre-Dashboard Integration Audit  
**Status**: AUDIT COMPLETE — FROZEN BASELINE VERIFIED  
**Date**: October 2026  

---

## 1. Executive Summary & Verification of Frozen Baseline

This audit formally inspects the repository architecture, data integrity, and legacy components before creating the production dashboard. In accordance with safety rules:
- **No code or research logic has been modified.**
- **No files have been deleted.**
- **No dashboard implementation has commenced.**
- **All frozen baseline values were verified directly from physical files on disk.**

### Master Master Verification of Frozen Research Results

Every single metric from the Step 10B and Step 11 frozen audit was loaded and verified directly from the canonical result files:

| Metric | TP53 | HBB | EGFR | BRCA1 | Master Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Candidates** (`gRNA_features.csv`) | 3,554 | 540 | 23,112 | 15,527 | **42,733** |
| **Biophysical / Censoring Excluded** | 1,906 | 175 | 7,022 | 9,749 | **18,852** |
| **Feasible Set** (`feasible_candidates.csv`) | 1,648 | 365 | 16,090 | 5,778 | **23,881** |
| **Pareto Front Size** (`pareto_front.csv`) | 8 | 10 | 8 | 10 | **36** |
| **Selected Top-5 Guides** (`selected_guides.csv`) | 5 | 5 | 5 | 5 | **20** |
| **Selected Guide IDs Match Baseline?** | **100% (5/5)** | **100% (5/5)** | **100% (5/5)** | **100% (5/5)** | **20 / 20** |

#### Verified Selected Guide Identifiers:
- **TP53**: `TP53_150_-`, `TP53_11911_-`, `TP53_559_-`, `TP53_2766_+`, `TP53_3053_-`
- **HBB**: `HBB_1533_+`, `HBB_3541_+`, `HBB_1202_-`, `HBB_1861_-`, `HBB_2265_-`
- **EGFR**: `EGFR_100693_+`, `EGFR_90787_-`, `EGFR_74407_+`, `EGFR_27703_-`, `EGFR_167075_+`
- **BRCA1**: `BRCA1_2088_+`, `BRCA1_2323_+`, `BRCA1_27410_-`, `BRCA1_122416_-`, `BRCA1_74210_+`

---

## 2. Canonical Computational Pipeline (13 Stages)

The complete end-to-end computational pipeline is structured into 13 execution stages:

```
[1. Target Gene / FASTA] ──> [2. Protospacer Extraction] ──> [3. Sequence Filtering]
                                                                      │
[6. DeepSpCas9 CNN] <── [5. RNA Secondary Structure] <── [4. Biophysical Modeling]
         │
         ▼
[7. Bowtie 1.3.1 Alignment (GRCh38)] ──> [8. CFD Scoring & Rescoring]
                                                    │
                                                    ▼
[11. Pareto Dominance] <── [10. Multi-Obj. LP] <── [9. Feasibility Filter]
         │
         ▼
[12. Sensitivity Audit] ──> [13. Final Balanced Top-5 Selection]
```

### Detailed Stage-by-Stage Specification

| Stage | Name | Script / Module | Input Artifact | Output Artifact | Key Dependencies |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | Target Sequence Resolution | `src/gene_sequence_provider.py`<br/>`src/ncbi_sequence_provider.py` | Gene symbol (e.g. `TP53`) or FASTA path | `data/reference_sequences/*.fasta` | Biopython, NCBI Entrez E-utilities |
| **2** | Candidate Protospacer Extraction | `src/guide_generation.py` | Target genomic FASTA | `results/<GENE>/candidate_gRNAs.csv` | Biopython, NGG PAM regex scanner |
| **3** | Sequence Filtering | `src/feature_extraction.py` | Protospacer table | Candidate subset (GC, homopolymer, poly-T) | NumPy, Pandas |
| **4** | Biophysical Modeling | `src/feature_extraction.py` | Protospacer sequences | Nearest-neighbor $T_m$ (SantaLucia), duplex $\Delta G$ | `Bio.SeqUtils.MeltingTemp` |
| **5** | Structural RNA Modeling | `src/feature_extraction.py` | Protospacer sequences | ViennaRNA secondary structure, MFE, paired base fraction | ViennaRNA (`RNA` module) |
| **6** | Deep Learning On-Target Prediction | `src/deep_spcas9.py` | 30bp genomic sequence context | `results/<GENE>/gRNA_features.csv` | PyTorch, `DeepCas9_weights.pt` |
| **7** | Off-Target Alignment | Bowtie 1.3.1 (WSL2) | Protospacer sequences | Alignment SAM/BAM hits against GRCh38.p14 | Bowtie 1.3.1, WSL2 / Linux |
| **8** | Cutting Frequency Determination | `src/cfd_score.py`<br/>`scripts/regenerate_step10a_rescored.py` | Bowtie alignments + reference | `results/step10a_cfd_rescored/cfd_guide_features.csv` | Doench mismatch & PAM matrices |
| **9** | CFD-Aware Feasibility Filtering | `scripts/step10b_optimize.py` | Feature table + CFD dataset | `results/step10b_optimization/<GENE>/run1/feasible_candidates.csv` | Pandas, `cfd_scoring_complete` |
| **10** | Multi-Objective Optimization | `scripts/step10b_optimize.py` | Feasible candidates | Track A LP continuous relaxation dual multipliers | CVXOPT (`cvxopt.solvers.lp`) |
| **11** | Pareto Dominance Filtering | `src/grna_optimizer.py` | Normalized Activity + CFD Specificity | `results/step10b_optimization/<GENE>/run1/pareto_front.csv` | Vectorized Pareto dominance, $\epsilon$-constraint |
| **12** | Sensitivity Analysis & Audit | `scripts/step11_sensitivity.py` | Step 10B outputs + CFD | `results/step11_sensitivity/step11_sensitivity_summary.json` | Adversarial censored imputation engine |
| **13** | Balanced Guide Selection | `scripts/step10b_optimize.py` | Non-dominated Pareto frontier | `results/step10b_optimization/<GENE>/run1/selected_guides.csv` | Balanced utility scalarization ($w_1=0.5, w_2=0.5$) |

---

## 3. Canonical Datasets for Dashboard Consumption

The production dashboard will ingest strictly from the canonical data artifacts:

1. **Gene Feature Tables**: `results/<GENE>/gRNA_features.csv`  
   - Source for raw candidate counts, biophysical properties ($T_m$, GC%, $\Delta G$, MFE), sequence strings, and DeepSpCas9 scores.
2. **Validated CFD Dataset**: `results/step10a_cfd_rescored/cfd_guide_features.csv`  
   - Source for cumulative off-target cutting risk ($R$), CFD specificity ($S = \frac{100}{100+R}$), scoring status, and scored off-target site counts.
3. **Validated Optimization Outputs**: `results/step10b_optimization/<GENE>/run1/`  
   - `feasible_candidates.csv`: Pre-filtered feasible pool with normalized scores.
   - `pareto_front.csv`: Non-dominated Pareto guides.
   - `selected_guides.csv`: Final Top-5 balanced therapeutic guide designs.
   - `optimization_summary.json`: High-level summary metrics, LP objective values, and feasibility funnel attrition numbers.
4. **Sensitivity & Audit Outputs**: `results/step11_sensitivity/`  
   - `step11_sensitivity_summary.json`: Detailed records of censored-guide worst-case scenarios, 50/100/200 $\epsilon$-step resolution checks, and rank correlation metrics ($\rho = 1.000000$).

---

## 4. Origin of Dashboard Metrics

The dashboard must display verified metrics derived directly from the canonical files without mock or hardcoded numbers:

| Dashboard Metric | Data Source File | Specific Column / Key | Transformation / Usage |
| :--- | :--- | :--- | :--- |
| **Total Candidates** | `results/<GENE>/gRNA_features.csv` | Row count `len(df)` | Displayed in KPI cards and attrition funnel |
| **Excluded Candidates** | `results/step10b_optimization/<GENE>/run1/optimization_summary.json` | `feasibility.excluded` | Breakdown into GC, homopolymer, poly-T, censored |
| **Feasible Candidates** | `results/step10b_optimization/<GENE>/run1/feasible_candidates.csv` | Row count `len(df)` | Plotted as background feasible cloud |
| **Pareto Front** | `results/step10b_optimization/<GENE>/run1/pareto_front.csv` | All rows | Plotted as blue non-dominated frontier curve |
| **Selected Top-5 Guides** | `results/step10b_optimization/<GENE>/run1/selected_guides.csv` | All rows ($k=5$) | Highlighted as red stars; detailed in summary table |
| **DeepSpCas9 Activity** | `gRNA_features.csv` / `feasible_candidates.csv` | `deep_spcas9_score` (raw) / `norm_activity` | Primary Y-axis on Pareto plots; candidate table |
| **CFD Specificity** | `cfd_guide_features.csv` / `feasible_candidates.csv` | `cfd_specificity_score` / `norm_cfd_specificity` | Primary X-axis on Pareto plots; candidate table |
| **CFD Off-Target Risk** | `cfd_guide_features.csv` / `feasible_candidates.csv` | `cfd_off_target_risk` | Cumulative risk sum $R = \sum \text{CFD}_j$ |
| **GC Content** | `gRNA_features.csv` | `gc_content` | Formatted as percentage ($30\% - 70\%$) |
| **Melting Temperature** | `gRNA_features.csv` | `melting_temp_c` | Nearest-neighbor $T_m$ in °C |
| **Duplex $\Delta G$** | `gRNA_features.csv` | `delta_g_kcal_mol` | Thermodynamic hybridization energy in kcal/mol |
| **Secondary Structure MFE** | `gRNA_features.csv` | `folding_energy_kcal_mol` | ViennaRNA folding free energy in kcal/mol |
| **Dot-Bracket Structure** | `gRNA_features.csv` | `secondary_structure` | Visualized in sequence inspector |
| **Off-Target Hit Counts** | `gRNA_features.csv` / `cfd_guide_features.csv` | `off_target_count`, `cfd_scored_site_count` | Mismatch distribution (0, 1, 2, 3 mismatches) |
| **Sensitivity Stability** | `results/step11_sensitivity/step11_sensitivity_summary.json` | `part2_censored_sensitivity`, `part3_epsilon_resolution` | Displayed in Robustness & Sensitivity tab |

---

## 5. Inspection of Existing Dashboard

### Findings from `dashboard/index.html` & `src/generate_dashboard.py`:
1. **Static / Hardcoded Step 5 Data**: `dashboard/index.html` is a 4.1 MB monolithic static HTML file generated by `src/generate_dashboard.py`. It does not make dynamic API calls; instead, it bakes an early 2,860-candidate TP53 dataset directly into JavaScript variables.
2. **Single-Gene Architecture**: There is no support or interface for switching between TP53, HBB, EGFR, and BRCA1.
3. **Absence of CFD Metrics**: It uses the obsolete unweighted heuristic $OT \le 20$ from Step 5, completely lacking the validated Cutting Frequency Determination metrics developed in Steps 10A/10B.
4. **Obsolete Objective Function**: It presents the deprecated 3-way scalar score ($0.35 E + 0.40 S + 0.25 R$ with Doench 2016 proxy) rather than DeepSpCas9 on-target activity and CFD specificity Pareto dominance.
5. **Runtime Server Error**: `src/run_dashboard.py` line 34 contains an unescaped newline syntax error (`print("\n`), causing the local web server to crash on launch.

**Conclusion**: The existing dashboard cannot be salvaged through minor edits. A modern, multi-gene dashboard with clean backend decoupling is required.

---

## 6. Pipeline Controller Status

- **Current Status**: `src/pipeline_controller.py` **does NOT exist** in the repository.
- **Action**: It must be designed and implemented to provide a clean, decoupled programmatic API that reads canonical results for the dashboard and exposes live execution capabilities.

---

## 7. Redundancy & Deprecation Classification

All files across the repository are categorized as follows:

| Classification | Meaning | Files Included |
| :--- | :--- | :--- |
| **ACTIVE** | Core components required by canonical pipeline and test suite | `main.py`, `src/gene_sequence_provider.py`, `src/ncbi_sequence_provider.py`, `src/guide_generation.py`, `src/feature_extraction.py`, `src/deep_spcas9.py`, `src/cfd_score.py`, `src/grna_optimizer.py`, `scripts/regenerate_step10a_rescored.py`, `scripts/step10b_optimize.py`, `scripts/step11_sensitivity.py`, `tests/run_all_tests.py`, `tests/test_cfd_score.py`, `tests/test_cfd_e2e.py`, `models/.../DeepCas9_weights.pt` |
| **LEGACY** | Superseded historical milestones preserved for auditability | `src/multi_gene_validation.py`, `src/step8_benchmarking.py`, `src/step9_cfd.py`, `src/step9c_cfd.py`, `scripts/regenerate_step9_final.py`, `src/convert_weights.py`, `src/inspect_checkpoint.py`, `results/step9_*`, `results/step8_benchmarking/`, `results/multi_gene_validation/` |
| **DEPRECATED** | Obsolete prototype implementations superseded by Step 10B/11 | `src/optimization.py`, `src/convex_optimization.py`, `src/visualization.py`, `src/convex_visualization.py`, `src/create_weighted_vs_convex_selection.py`, `src/build_technical_pdf.py`, `dashboard/create_bio_backend_pdf.py`, `dashboard/index.html`, `src/generate_dashboard.py`, `src/run_dashboard.py` |
| **ARCHIVE-CANDIDATE** | Unneeded clutter safe to move or delete during future cleanup | `src.zip` (root archive), `data/gRNA_features_specificity.csv`, `data/optimized_guides.csv`, `data/convex_optimized_guides.csv` |
| **UNKNOWN** | External third-party submodules retained without modification | `models/Paired-Library/DeepBaseEditor/`, `DeepCpf1/`, `DeepCas9-NG/`, `DeepxCas9/` |

---

## 8. Discrepancy Analysis

- **Investigation**: We cross-checked the frozen baseline numbers against all raw files on disk.
- **Result**: **ZERO physical discrepancies exist.**
  - `results/<GENE>/gRNA_features.csv` has exactly 3,554 (TP53), 540 (HBB), 23,112 (EGFR), and 15,527 (BRCA1) rows.
  - `results/step10a_cfd_rescored/cfd_guide_features.csv` deduplicates to exactly 42,733 unique guide IDs, matching the feature files 1:1.
  - `results/step10b_optimization/` feasible candidate counts are exactly 1,648 (TP53), 365 (HBB), 16,090 (EGFR), and 5,778 (BRCA1).
  - The alternative numbers appearing in earlier draft notes (3,363; 467; 24,082; 27,272) were confirmed to be typographical drafting artifacts that do not exist on disk. All reports have been aligned with the physical data.

---

## 9. Recommended Architecture for `src/pipeline_controller.py`

To strictly prevent coupling between the frontend UI and research algorithms, `src/pipeline_controller.py` will serve as the unified bridge:

```python
"""
src/pipeline_controller.py

Unified Backend Controller for CRISPR Guide RNA Optimization Platform.
Provides structured, decoupled data retrieval and live pipeline orchestration.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import json

class PipelineController:
    """
    Decoupled interface mediating between data stores, computational engines,
    and user-facing dashboards or APIs.
    """
    
    @classmethod
    def get_supported_genes(cls) -> List[str]:
        """Returns list of pre-computed validated genes ['TP53', 'HBB', 'EGFR', 'BRCA1']."""
        ...

    @classmethod
    def load_gene_overview(cls, gene: str) -> Dict[str, Any]:
        """
        Loads metadata, candidate counts, feasibility funnel statistics,
        and objective ranges from canonical optimization summaries.
        """
        ...

    @classmethod
    def load_pareto_front(cls, gene: str) -> pd.DataFrame:
        """Loads verified non-dominated Pareto frontier candidates."""
        ...

    @classmethod
    def load_feasible_candidates(cls, gene: str, sample_limit: Optional[int] = None) -> pd.DataFrame:
        """Loads feasible candidates with normalized scores for scatter plotting."""
        ...

    @classmethod
    def load_selected_guides(cls, gene: str) -> pd.DataFrame:
        """Loads Top-5 balanced guide selections with comprehensive biophysical annotations."""
        ...

    @classmethod
    def load_sensitivity_summary(cls, gene: str) -> Dict[str, Any]:
        """Loads censored worst-case and epsilon-resolution audit metrics for gene."""
        ...

    @classmethod
    def run_live_pipeline(cls, sequence_or_fasta: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes candidate generation, feature extraction, DeepSpCas9 prediction,
        and optimization for custom user inputs.
        """
        ...
```

---

## 10. Exact Next Implementation Steps (Awaiting Approval)

Once approved by the user, implementation will proceed through the following non-destructive steps:

1. **Step 1: Implement `src/pipeline_controller.py`**
   - Build and test the decoupled controller reading from `results/step10b_optimization/`, `results/step11_sensitivity/`, and `results/<GENE>/`.
   - Validate that it returns exact frozen counts and guide IDs for all four genes.
2. **Step 2: Construct Production Multi-Gene Dashboard**
   - Build a research-grade, modern responsive dashboard in `dashboard/` (using HTML5/Vanilla CSS/Modern JS with Plotly.js for interactive Pareto charts).
   - Ensure complete multi-gene switching across **TP53**, **HBB**, **EGFR**, and **BRCA1**.
   - Implement tabs:
     - *Overview & Gene Master Benchmark*
     - *Interactive Pareto Frontier & Trade-off Explorer*
     - *Candidate Filter Funnel & Biophysical Metrics*
     - *Guide Design Table & Sequence Inspector*
     - *Sensitivity & Robustness Audit View*
     - *Custom Target Analysis (Live/Demo Mode)*
3. **Step 3: Build Robust Local Server**
   - Create `src/serve_dashboard.py` (replacing the broken `src/run_dashboard.py`) with clean REST endpoints or static JSON payload generation for standalone execution.
4. **Step 4: End-to-End Validation**
   - Verify in browser that all charts, tables, metrics, and filters function flawlessly and match the frozen baseline with zero errors.
