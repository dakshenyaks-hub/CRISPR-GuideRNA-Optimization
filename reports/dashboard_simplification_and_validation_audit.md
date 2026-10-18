# CRISPR Guide RNA Optimization: Navigation Simplification & Research Audit Report

**Document**: Formal Research Audit & Dashboard Simplification Report  
**Project**: A Multi-Objective Optimization Framework for CRISPR Guide RNA Design Using Machine Learning and Mathematical Optimization  
**Auditor**: Senior Software Engineer, Research Auditor, & Scientific Dashboard Designer  
**Status**: AUDIT COMPLETE — FROZEN BASELINE PRESERVED — ALL TESTS PASSED  
**Date**: October 2026  

---

## 1. Executive Summary

This audit addresses navigation complexity and data-integrity discrepancies in the CRISPR Guide RNA Optimization platform. In strict compliance with research safety requirements:
1. **Zero research algorithms or pipeline scripts were altered**: Steps 1–11, CFD scoring matrices, PyTorch DeepSpCas9 CNN models, and multi-objective Pareto optimization code remain intact and unmodified.
2. **Zero research datasets were deleted or overwritten**: Historical files (Steps 7, 8, 9, 9C, 10A, 10B, 11) were treated as read-only.
3. **The dashboard navigation was simplified from 8 tabs to 5 clear, high-level tabs**:
   - **Lab** (`🧪 LAB`): Central interactive sequential engineering workbench.
   - **CRISPR Mechanism** (`🧬 CRISPR MECHANISM`): Interactive Cas9 visualizer coupled with biophysical foundations.
   - **Analysis** (`📊 ANALYSIS`): Progressive biophysical and off-target distribution charts.
   - **Mathematical Formulation** (`📐 MATHEMATICAL FORMULATION`): Dual-objective mathematical modeling, continuous LP relaxation, interactive optimization lab, and practical guidance FAQ.
   - **Validation & Robustness** (`⚖️ VALIDATION & ROBUSTNESS`): Unified cross-gene validation benchmark (Section A) and multi-scenario robustness & sensitivity audit (Section B).
4. **All data discrepancies identified in prior dashboard versions were investigated directly to their raw data origins**, resolved through independent mathematical recalculation, and corrected in the presentation layer with transparent status labels (`VERIFIED`, `CHANGED`, `UNVERIFIED`, `FAILED`).

---

## 2. Mandatory Data-Integrity Investigation

Before modifying dashboard views, all reported numerical discrepancies were traced through raw CSVs, intermediate pipeline runs, and dashboard data-generation scripts.

| Discrepancy | Previous Display | Authoritative Baseline | Root Cause Diagnosis | Evidence Files | Final Status |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **Total Candidates Count** | 45,337 | **42,733** | **Stale un-deduplicated / mixed-version data**. In Step 10A, raw alignment tables contained duplicate entries from a many-to-many merge in `step9c_cfd.py` (59,361 raw rows with 16,628 duplicates). Deduplication on `guide_id` yielded 42,733 unique protospacers across the 4 validated genes. An intermediate benchmark script reported an un-deduplicated figure of 45,337. | `results/<GENE>/gRNA_features.csv`<br/>`results/step10b_optimization/step10b_optimization_summary.json` (Line 9) | **VERIFIED** (Corrected to 42,733) |
| **Feasible Pool Total** | 25,066 | **23,881** | **Coupled with TP53 pre-CFD filtering discrepancy**. When Bowtie capped guides ($CFD = \text{NaN}$) were excluded in Step 10B, TP53 feasible designs dropped from ~2,833 to 1,648 (difference = 1,185). An earlier dashboard table combined the old TP53 count (2,833) with Step 10B counts for HBB (365), EGFR (16,090), and BRCA1 (5,778), summing to exactly 25,066 (23,881 + 1,185). | `results/step10b_optimization/<GENE>/run1/feasible_candidates.csv`<br/>`results/TP53/optimization/feasible_candidates.csv` | **VERIFIED** (Corrected to 23,881) |
| **TP53 Feasible Count** | 2,833 | **1,648** | **Application of CFD search completeness filter**. Prior to Step 10B (e.g. Step 7/8), feasible filtering only evaluated biophysical constraints (GC, poly-T, homopolymers), leaving ~2,791–2,833 candidates. Step 10B strictly excluded 1,143 capped guides with truncated Bowtie alignments and unquantifiable risk, leaving exactly 1,648 feasible candidates. | `results/step10b_optimization/TP53/run1/feasible_candidates.csv` (1,648 rows)<br/>`excluded_candidates.csv` (1,143 capped) | **VERIFIED** (Corrected to 1,648) |
| **Gene Scope (KRAS & RB1)** | 6 genes shown | **4 validated genes** | **Exploratory test sequences included in error**. `KRAS` (4,329 candidates) and `RB1` (15,608 candidates) FASTA sequences were fetched into `data/reference_sequences/` during early development of the NCBI sequence provider. They were never aligned across GRCh38 via Bowtie or scored with CFD. Only TP53, HBB, EGFR, and BRCA1 were subjected to complete experimental validation. | `data/reference_sequences/KRAS_NC_000012_12.fasta`<br/>`src/multi_gene_validation.py` (Lines 57–90) | **VERIFIED** (Scope restricted to 4 core genes) |

### Canonical Four-Gene Authoritative Benchmark

Independent recalculation directly from `results/<GENE>/gRNA_features.csv`, `feasible_candidates.csv`, `pareto_front.csv`, and `selected_guides.csv` confirms the exact canonical baseline:

$$\begin{array}{l|rrrrrr}
\textbf{Target Gene} & \textbf{Total Candidates} & \textbf{Excluded} & \textbf{Feasible Pool} & \textbf{Yield (\%)} & \textbf{Pareto Front} & \textbf{Selected (}k=5\textbf{)} \\
\hline
\text{TP53} & 3,554 & 1,906 & 1,648 & 46.4\% & 8 & 5 \\
\text{HBB} & 540 & 175 & 365 & 67.6\% & 10 & 5 \\
\text{EGFR} & 23,112 & 7,022 & 16,090 & 69.6\% & 8 & 5 \\
\text{BRCA1} & 15,527 & 9,749 & 5,778 & 37.2\% & 10 & 5 \\
\hline
\textbf{MASTER TOTAL} & \mathbf{42,733} & \mathbf{18,852} & \mathbf{23,881} & \mathbf{55.9\%} & \mathbf{36} & \mathbf{20}
\end{array}$$

---

## 3. Audit of All 100% and Invariance Claims

All claims of "100%" or zero-delta invariance were audited against the underlying implementation and raw data.

### 1. "100% Baseline Agreement"
- **Claim**: Top-5 selected guide IDs match the frozen baseline 100% across all 4 genes.
- **Investigation**: Verified against `results/step10b_optimization/<GENE>/run1/selected_guides.csv` and `run2/selected_guides.csv`. The 20 guide IDs (5 per gene) are identical across runs:
  - **TP53**: `TP53_150_-`, `TP53_11911_-`, `TP53_559_-`, `TP53_2766_+`, `TP53_3053_-`
  - **HBB**: `HBB_1533_+`, `HBB_3541_+`, `HBB_1202_-`, `HBB_1861_-`, `HBB_2265_-`
  - **EGFR**: `EGFR_100693_+`, `EGFR_90787_-`, `EGFR_74407_+`, `EGFR_27703_-`, `EGFR_167075_+`
  - **BRCA1**: `BRCA1_2088_+`, `BRCA1_2323_+`, `BRCA1_27410_-`, `BRCA1_122416_-`, `BRCA1_74210_+`
- **Audit Verdict**: **VERIFIED**.

### 2. "100% Top-Five Selection Overlap Under Censored Imputation"
- **Claim**: Top-5 guide selections are 100% invariant when censored alignment searches are imputed with worst-case off-target risk.
- **Investigation**: In Step 11 (`results/step11_sensitivity/censored_worst_case/`), all censored candidates (1,589 in TP53, 124 in HBB, 3,912 in EGFR, 8,423 in BRCA1) were assigned the gene-specific maximum observed risk $R_{\text{max}}$.
- **Mathematical Cause**: Imputing worst-case risk degrades specificity to $S_i = \frac{100}{100 + R_{\text{max}}} \le 0.306$. Because the primary Top-5 guides have high specificity ($S_i \ge 0.95$) and high activity ($E_i \ge 75$), penalizing capped candidates drives them to the bottom of the ranking. They cannot outcompete the uncapped designs.
- **Audit Verdict**: **VERIFIED** for Top-5 portfolio stability. However, this measures stability under worst-case penalty, **not universal off-target safety**.

### 3. "BRCA1 Pareto Front Size Under Imputation" (Previously Suppressed Shift)
- **Previous Claim**: Pareto front size remained 10 across both primary and imputed runs.
- **Investigation**: Physical inspection of `results/step11_sensitivity/step11_sensitivity_summary.json` (Line 353) and `results/step11_sensitivity/censored_worst_case/BRCA1/run1/pareto_front.csv` reveals that the BRCA1 Pareto front expanded from **10 to 11 guides**. An imputed guide with high on-target cleavage (83.27%) and worst-case specificity (0.1796) established a new boundary trade-off point.
- **Correction**: The dashboard previously masked this by hardcoding `meta.pareto` (10) in both columns. It has been updated to honestly report **10 &rarr; 11** with the status badge **CHANGED**.
- **Audit Verdict**: **CHANGED** (Transparently reported).

### 4. "Spearman Correlation of 1.00000"
- **Previous Claim**: 1.00000 Spearman rank correlation proved model stability under perturbation.
- **Investigation**: In `scripts/step11_sensitivity.py` (Line 108), the correlation was computed as:
  ```python
  rank_spec = g["cfd_specificity_score"].rank(ascending=False)
  rank_risk = g["cfd_off_target_risk"].rank(ascending=True)
  corr = float(rank_spec.corr(rank_risk, method="spearman"))
  ```
  Since $S = \frac{100}{100 + R}$ is a strictly decreasing monotonic transformation, $\text{rank}(S) \equiv -\text{rank}(R)$. The correlation is a **mathematical tautology**, not an empirical model stability test.
- **Correction**: The dashboard separates this into:
  - *Formula Monotonicity Identity*: Verified formula property ($\rho \equiv 1.0000$), labelled as a mathematical sanity check.
  - *Top-5 Portfolio Monotonicity*: Zero rank substitutions ($1 \to 5$) between primary and imputed selections (Verified).
- **Audit Verdict**: **VERIFIED AS FORMULA IDENTITY**, previously misrepresented as model robustness.

### 5. "Zero Pareto-Front Change Across Epsilon Resolutions ($\Delta = 0$)"
- **Claim**: The Pareto front size is identical whether solved with 50, 100, or 200 $\epsilon$-constraint intervals.
- **Investigation**: Verified across `results/step11_sensitivity/epsilon_resolution/`. Because the 2D non-dominated set consists of 8–10 discrete points, a 50-step grid already samples between all adjacent points. Increasing to 100 or 200 steps discovers no additional points.
- **Audit Verdict**: **VERIFIED**.

---

## 4. Summary of Dashboard File Modifications

All modifications were confined to the presentation and testing layers:

```
dashboard/
└── index.html                    [MODIFIED: Navigation simplified to 5 tabs;
                                   Biology biophysical foundations merged into CRISPR Mechanism;
                                   Interpretation FAQs merged into Mathematical Formulation;
                                   Combined Validation & Robustness page built;
                                   Honest status badges and dynamic calculations added]

tests/
├── test_validation_dashboard.py   [CREATED: Automated test suite verifying 5-tab structure,
                                   content relocation, 4-gene authoritative counts, and honest statuses]
└── run_all_tests.py              [MODIFIED: Registered test_validation_dashboard in master suite]
```

### Research Code and Data Integrity Preservation
- `main.py`: Unmodified.
- `src/cfd_score.py`: Unmodified.
- `src/grna_optimizer.py`: Unmodified.
- `src/deep_spcas9.py`: Unmodified.
- `scripts/step10b_optimize.py`: Unmodified.
- `scripts/step11_sensitivity.py`: Unmodified.
- `results/`: Read-only. All canonical files preserved.

---

## 5. Verification Commands and Test Results

The test suite was executed via the master test runner:

```powershell
python tests/run_all_tests.py
```

### Test Execution Output:
```
=== Running Unit Tests in tests/test_cfd_score.py ===
  [PASS] TestMismatchTable
  [PASS] TestPAMTable
  [PASS] TestCalcCFDSite
  [PASS] TestPAMKeyExtraction
  [PASS] TestIntendedLocusExclusion
  [PASS] TestPerfectMatchRetention
  [PASS] TestCensoredHandling
  [PASS] TestZeroHitDistinction
  [PASS] TestNoRawCountSubstitution
  [PASS] TestUnsupportedPAM
  [PASS] TestSchemaPreservation
  [PASS] TestReproducibility
  [PASS] TestStep10AVarianceAndExclusionRegression

=== Running E2E Tests in tests/test_cfd_e2e.py ===
  [PASS] test_actual_grch38_extraction
  [PASS] test_exclude_on_target_unresolved

=== Running Pipeline Controller Baseline Tests in tests/test_pipeline_controller.py ===
  [PASS] test_supported_genes
  [PASS] test_frozen_baseline_integrity (All 4 genes matched)
  [PASS] test_gene_overview_metrics for TP53, HBB, EGFR, BRCA1
  [PASS] test_pareto_front_records for TP53, HBB, EGFR, BRCA1
  [PASS] test_selected_guides_records for TP53, HBB, EGFR, BRCA1
  [PASS] test_sensitivity_summary for TP53, HBB, EGFR, BRCA1
  [PASS] test_full_payload_generation
  [PASS] PipelineController 100% frozen baseline fidelity

=== Running NCBI Sequence Provider Tests in tests/test_ncbi_sequence_provider.py ===
  [PASS] TestGeneSymbolValidation
  [PASS] TestDataStructures.test_metadata_to_dict

=== Running Mathematical Formulation & Optimization Lab Tests in tests/test_optimization_lab.py ===
  [PASS] TestOptimizationLabStructure (KaTeX, Sequence, DOM elements)
  [PASS] TestOptimizationLabCalculations (Weight sweeps, Baseline fidelity, Pareto optimality)

=== Running Validation & Robustness Dashboard Tests in tests/test_validation_dashboard.py ===
  [PASS] TestDashboardNavigationAndStructure (5 Tabs, Content Relocation, Section A & B)
  [PASS] TestValidationDataIntegrity (4-Gene Authoritative Baseline & Honest Statuses)

ALL UNIT, E2E, PIPELINE CONTROLLER, NCBI PROVIDER, OPTIMIZATION LAB, AND VALIDATION DASHBOARD TESTS PASSED SUCCESSFULLY!
```

---

## 6. Audit Conclusion & Recommendations

1. **Dashboard Usability**: The platform now features a clean 5-tab structure that eliminates navigational redundancy while preserving 100% of the biophysical, mathematical, and practical explanations.
2. **Scientific Honesty**: The dashboard explicitly presents data boundaries, distinguishes mathematical formula tautologies from empirical robustness, honestly flags variations (such as the BRCA1 Pareto shift), and reinforces that in silico mathematical optimization requires experimental wet-lab validation prior to clinical application.
