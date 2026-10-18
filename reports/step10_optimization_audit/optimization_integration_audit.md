# Step 10 — Downstream Optimization Integration and Scientific Validation Audit Report

## 1. Executive Summary
This audit evaluated the downstream mathematical optimization pipeline in relation to the newly generated Step 9C CFD specificity dataset (`results/step9_cfd_final/cfd_guide_features.csv`). 

### Core Audit Discoveries:
1. **Pipeline Integration Disconnect:** The downstream optimization modules (`src/convex_optimization.py`, `src/optimization.py`, and `src/grna_optimizer.py`) are currently **disconnected from Step 9C**. They do not load `results/step9_cfd_final/cfd_guide_features.csv`. Instead, `convex_optimization.py` hard-codes `data/gRNA_features_specificity.csv` (a 2,860-guide legacy single-gene table), while `grna_optimizer.py` loads `results/{gene}/gRNA_features.csv` (the Step 7 raw feature table). Neither module ingests `cfd_specificity_score` or `cfd_off_target_risk`.
2. **Mystery of the 4,457 "Suspicious Guides" Solved:** Forensic examination of the historical `results/step9_cfd/cfd_guide_features.csv` revealed that exactly **4,457 guides** had `cfd_off_target_risk == 999.0000`. This occurred because the obsolete Step 9 implementation directly substituted the raw Bowtie hit cap ($999$) as the numeric CFD risk value, creating artificial spikes at $999.0$. In the regenerated `step9_cfd_final` dataset, this defect is completely resolved: all 30,062 capped guides are strictly set to `NaN`.
3. **Variance Collapse in Final CFD Scores:** In `results/step9_cfd_final/cfd_guide_features.csv`, all 29,299 uncapped guides have identical values: `cfd_off_target_risk = 1.0` and `cfd_specificity_score = 0.990099`. This occurred because the upstream `step9_cfd_corrected` table had empty site counts for uncapped guides, and the regeneration script added $+1.0$ risk to represent the unresolved on-target hit ($0.0 + 1.0 = 1.0$). Consequently, the current CFD specificity column has zero variance across uncapped guides, causing any Pareto front constructed against CFD specificity to collapse to a single point.
4. **Feasibility Asymmetry in Convex Optimization:** Under `convex_optimization.py`'s constraint of `off_target_count <= 20`, only **45 guides** are feasible across all four genes, with **0 feasible guides for HBB** due to globin pseudogene homologous hits. In contrast, under `grna_optimizer.py` biological filters, **24,431 guides** are feasible.

**Overall Audit Finding:** **NOT READY FOR DOWNSTREAM OPTIMIZATION** (until optimizer inputs are re-pointed to Step 9C and the zero-variance CFD scoring anomaly is resolved).

---

## 2. File and Path Verification

| File / Component | Verified Path | Current Data Input | Intended Data Input | Status |
|---|---|---|---|---|
| **Convex LP Optimizer** | `src/convex_optimization.py` | `data/gRNA_features_specificity.csv` (2,860 rows) | `results/step9_cfd_final/cfd_guide_features.csv` | **MISALIGNED** |
| **Weighted Sum Optimizer** | `src/optimization.py` | `data/gRNA_features_specificity.csv` (2,860 rows) | `results/step9_cfd_final/cfd_guide_features.csv` | **MISALIGNED** |
| **Multi-Objective Optimizer** | `src/grna_optimizer.py` | `results/{gene}/gRNA_features.csv` | `results/step9_cfd_final/cfd_guide_features.csv` | **MISALIGNED** |
| **Multi-Gene Benchmark** | `src/step8_benchmarking.py` | `results/{gene}/optimization/*.csv` | `results/step9_cfd_final/` | Preserved |
| **Step 9C Final Dataset** | `results/step9_cfd_final/cfd_guide_features.csv` | N/A (Output of Step 9C) | 59,361 guides (4 genes) | Verified on disk |

---

## 3. Candidate Eligibility & Feasibility Audit

### 3.1 Capped and Incomplete Alignment Searches
- **Dataset Total:** 59,361 guides.
- **Capped Searches (`off_target_capped == True`):** 30,062 guides (50.64%).
  - In `results/step9_cfd_final/cfd_guide_features.csv`, all 30,062 guides are strictly classified as `cfd_scoring_complete = False`, `cfd_scoring_status = "CENSORED_CAP_REACHED"`, with `cfd_off_target_risk = NaN` and `cfd_specificity_score = NaN`.
  - In `src/convex_optimization.py` (line 276), `capped_ok = ~df["off_target_capped"]` correctly ensures that no capped guide can enter the feasible set ($u_i = 0$).
  - In `src/grna_optimizer.py` (lines 136–141), `capped_guide_policy` defaults to `"retain"` (treating capped guides as lower-bound counts), which would allow guides with `NaN` CFD scores into the pool unless set to `"exclude"`.

### 3.2 Missing Scores
- Across the 29,299 complete guides, exactly **4 guides** have missing (`NaN`) `deep_spcas9_score`:
  - `TP53_0_-`, `EGFR_196607_+`, `EGFR_196608_+`, `BRCA1_22_+`.
- **Cause:** These guides lie at the sequence boundaries of their respective target FASTA files and lack the required 30-nt context (4 bp 5' flanking + 20 bp guide + 3 bp PAM + 3 bp 3' flanking) required by the DeepSpCas9 CNN.
- **Handling:** `src/grna_optimizer.py` (line 105) explicitly filters `df["deep_spcas9_score"].isna()`, eliminating these 4 records from the feasible set.

### 3.3 Unresolved Intended-Target Exclusions
- In `results/step9_cfd_final/cfd_guide_features.csv`, all 59,361 guides have `on_target_exclusion_method = "UNRESOLVED"` and `on_target_excluded = False`.
- **Scientific Implication:** Because relative coordinates in the guide table could not be mapped to chromosome-level NCBI accession spans in batch, the pipeline conservatively retained all 0-mismatch alignments. This safely prevents false exclusions of genuine multi-copy off-targets. However, it added $+1.0$ risk to every guide, artificially shifting the baseline risk of zero-off-target guides from $0.0$ to $1.0$.

### 3.4 Feasibility Filter Comparison

| Filter / Constraint | `src/convex_optimization.py` | `src/grna_optimizer.py` |
|---|---|---|
| **GC Content** | $[0.40, 0.60]$ | $[0.30, 0.70]$ |
| **Melting Temperature (Tm)** | $[55.0, 65.0]^\circ\text{C}$ | Not constrained |
| **Poly-T Terminator (`TTTT`)** | Not constrained | Must be `False` |
| **Homopolymer Run** | Not constrained | $\le 4$ nt |
| **Off-Target Limit** | $\text{off\_target\_count} \le 20$ | Optimized continuously |
| **Hit Cap** | Must be `False` | Policy configurable (`retain`/`exclude`) |
| **Total Feasible (Step 9C Final)** | **45 guides** | **24,431 guides** |
| - TP53 | 10 | 1,648 |
| - HBB | **0 (Empty set)** | 365 |
| - EGFR | 29 | 16,422 |
| - BRCA1 | 6 | 5,996 |

> **Critical Note on HBB:** In `convex_optimization.py`, the hard constraint $\text{off\_target\_count} \le 20$ causes the feasible set for HBB to be **empty** ($0$ guides). The globin gene family contains extensive homology (HBD, HBA1, HBA2, etc.), meaning virtually all HBB-targeting guides have $>20$ genomic hits. The linear program fails with `RuntimeError: Feasible set is EMPTY` if run on HBB.

---

## 4. Mathematical Formulation Audit

### 4.1 Decision Variables
- **`src/convex_optimization.py` (Track A):**
  - Continuous selection variable $x_i \in [0, 1]$ representing the relaxed selection level of candidate guide $i \in \{1, \dots, n\}$.
  - Bounded by box constraints $0 \le x_i \le u_i$, where $u_i \in \{0, 1\}$ is the biological feasibility indicator.
  - Due to the linear objective and constraint matrix structure (polytope vertices), the LP relaxation attains optimal values at extreme points ($x_i^* \in \{0, 1\}$), making the continuous relaxation exact for top-$K$ selection.
- **`src/grna_optimizer.py` (Track B):**
  - Discrete binary decision variable $x_i \in \{0, 1\}$.
  - $x_i = 1$ if guide $i$ is selected, $0$ otherwise.

### 4.2 Objective Functions & Units

#### Track A: Scalarized Linear Objective (`convex_optimization.py`)
$$\max_{\mathbf{x}} \sum_{i=1}^n c_i x_i$$
where the utility coefficient $c_i$ is a convex combination:
$$c_i = w_E E_i + w_S S_i + w_R R_i \quad \in [0, 1]$$
- **Weights:** $w_E = 0.35$ (Efficiency), $w_S = 0.40$ (Specificity), $w_R = 0.25$ (Structural suitability). $\sum w = 1.0$.
- **Efficiency ($E_i \in [0, 1]$):** Min-max normalized `doench2016_proxy_score` (dimensionless, maximized).
- **Specificity ($S_i \in [0, 1]$):** 
  $$S_i = 0.70 \cdot (1 - \hat{O}_i) + 0.30 \cdot \hat{M}_i$$
  where $\hat{O}_i$ is min-max normalized `off_target_count` and $\hat{M}_i$ is min-max normalized `mismatch_score`. (Dimensionless, maximized).
- **Structural Suitability ($R_i \in [0, 1]$):** Average of Derringer-Suich Tm desirability, inverted duplex $\Delta G$, and normalized self-folding MFE. (Dimensionless, maximized).

#### Track B: Bi-Criterion Vector Objective (`grna_optimizer.py`)
$$\max_{\mathbf{x}} \begin{pmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \end{pmatrix} = \begin{pmatrix} \sum_{i=1}^n \hat{E}_i x_i \\ \sum_{i=1}^n \hat{S}_i x_i \end{pmatrix}$$
- **$f_1$ (On-Target Activity):** Min-max normalized DeepSpCas9 CNN prediction score $\hat{E}_i \in [0, 1]$ (dimensionless, maximized).
- **$f_2$ (Specificity):** $\hat{S}_i = 1 - \hat{O}_i \in [0, 1]$ based on raw off-target count (dimensionless, maximized).
- **Balanced Scalarization (Tie-Breaking / Benchmark):**
  $$f_{\text{balanced}}(\mathbf{x}) = 0.5 f_1(\mathbf{x}) + 0.5 f_2(\mathbf{x})$$

#### Step 9C CFD Specificity Formulation (Pending Integration)
- **Site-Level Risk:** $CFD_{\text{site}} = \left(\prod \text{Mismatch Penalty}\right) \times \text{PAM Penalty} \in [0, 1]$.
- **Guide-Level Cumulative Risk:** $R_{\text{CFD}}(g) = \sum_{\text{loci}} CFD_{\text{site}} \in [0, \infty)$ (unbounded, minimized).
- **Normalized Specificity Convention (CRISPOR):**
  $$S_{\text{CFD}}(g) = \frac{100}{100 + R_{\text{CFD}}(g)} \in (0, 1] \quad (\text{dimensionless, maximized})$$
- **Mathematical Distinction:** $R_{\text{CFD}}$ represents expected off-target cleavage events (higher = worse). $S_{\text{CFD}}$ is a monotonically decreasing non-linear transformation mapping risk into a bounded efficiency-compatible scale (higher = better). They must never be added directly or treated as equivalent.

### 4.3 Constraints
1. **Biological Feasibility:** $x_i \le u_i \quad \forall i \in \{1, \dots, n\}$, where $u_i = 1$ if biological criteria pass, $0$ otherwise.
2. **Cardinality / Budget:** $\sum_{i=1}^n x_i \le K$ (where $K=10$ in convex LP, $k=1$ in Pareto selection).
3. **Domain:** $0 \le x_i \le 1$ (relaxed LP) or $x_i \in \{0, 1\}$ (discrete).

### 4.4 Convexity Analysis
- **`convex_optimization.py` Formulation:**
  - **Objective Function:** $\mathbf{c}^T \mathbf{x}$ is linear in $\mathbf{x}$, which is simultaneously convex and concave.
  - **Feasible Set:** Defined by $2n$ bound inequalities ($-x_i \le 0$, $x_i \le u_i$) and 1 budget inequality ($\sum x_i \le K$). Each inequality defines an affine halfspace $H_j = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{a}_j^T \mathbf{x} \le b_j\}$. The intersection of a finite number of affine halfspaces forms a closed, bounded convex polyhedron (polytope).
  - **Classification:** **Linear Program (Convex Optimization Problem)**. Solvable in polynomial time via CVXOPT's interior-point solver (`solvers.lp`).
- **`grna_optimizer.py` (Track B Formulation):**
  - Due to discrete integer constraints $x_i \in \{0, 1\}$, the feasible set is a finite set of vertices, which is **non-convex**. Solved via exact multi-objective dominance sorting and $\epsilon$-constraint scalarization.

---

## 5. Discovered CFD Anomaly: Variance Collapse in Step 9C Final

A critical mathematical issue was identified when attempting to construct a Pareto front with `results/step9_cfd_final/cfd_guide_features.csv`:
```
TP53 Feasible Candidates: 1,648
Old Pareto Front Size: 9
CFD Pareto Front Size: 1 (Front collapsed to a single point)
Pearson Correlation (Old Spec vs CFD Spec): -0.0000
Spearman Rank Correlation: NaN (Constant input array)
```
- **Reason:** Every complete guide in `results/step9_cfd_final/cfd_guide_features.csv` has `cfd_off_target_risk = 1.0` and `cfd_specificity_score = 0.990099`.
- **Impact on Optimization:** When an objective column is constant across all candidates, every candidate has identical specificity ($0.990099$). Consequently, multi-objective Pareto optimization collapses to single-objective maximization of `deep_spcas9_score`. The trade-off surface between on-target activity and off-target specificity cannot be evaluated.

---

## 6. Recommendations & Integration Plan

Before Step 10 downstream optimization can be declared complete, two specific engineering fixes are required:
1. **Restore Continuous CFD Risk Distribution:** Re-populate the uncapped guides' site-level CFD scores from the validated Bowtie alignment records so that `cfd_specificity_score` reflects genuine genomic off-target divergence rather than a uniform constant.
2. **Update Optimizer Input Routing:** Update `src/convex_optimization.py` and `src/grna_optimizer.py` to:
   - Accept `results/step9_cfd_final/cfd_guide_features.csv` as input.
   - Use `cfd_specificity_score` as the specificity objective.
   - Exclude `off_target_capped == True` guides.
   - Relax `off_target_count <= 20` for paralogous genes like HBB (or replace it with a threshold on $R_{\text{CFD}}$).

---

## Final Status

`NOT READY FOR DOWNSTREAM OPTIMIZATION`

*(Integration code is currently mispointed to legacy datasets, and the final CFD specificity scores in the regenerated table require score variance restoration before a valid Pareto front can be generated.)*
