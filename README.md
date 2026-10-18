# CRISPR Guide RNA Optimization Framework

**A Multi-Objective Optimization Framework for CRISPR-Cas9 Guide RNA Design Using Deep Learning and Biophysical Modeling**

---

## 1. Executive Summary

This repository implements a rigorous, end-to-end computational framework for designing and optimizing single-guide RNAs (sgRNAs) targeting human genes. The pipeline integrates dynamic genomic data acquisition via NCBI Entrez, biophysical sequence feasibility constraints, deep learning-based on-target cleavage prediction (**DeepSpCas9** CNN), genome-wide off-target cleavage risk evaluation (**Bowtie + Cutting Frequency Determination [CFD]**), and multi-objective Pareto optimization.

### Authoritative Four-Gene Research Baseline

All experimental findings, model evaluations, and dashboard visualizations are validated against an immutable, frozen baseline across four core disease-relevant target genes:

| Target Gene | Chromosome Window (GRCh38) | Clinical / Biological Relevance | Total Candidates | Excluded Candidates | Feasible Pool | Pareto Front | Selected Portfolio ($k=5$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TP53** | chr17:7,666,421–7,689,490 (−) | Cellular tumor antigen p53 | 3,554 | 1,906 | 1,648 | 8 | 5 |
| **HBB** | chr11:5,223,464–5,231,069 (−) | Hemoglobin subunit beta | 540 | 175 | 365 | 10 | 5 |
| **EGFR** | chr7:55,017,016–55,211,628 (+) | Epidermal growth factor receptor | 23,112 | 7,022 | 16,090 | 8 | 5 |
| **BRCA1** | chr17:43,042,293–43,172,000 (−) | Breast/ovarian cancer susceptibility | 15,527 | 9,749 | 5,778 | 10 | 5 |
| **MASTER TOTAL** | — | — | **42,733** | **18,852** | **23,881** | **36** | **20** |

#### Top-5 Selected Guide Portfolios ($w_1 = 0.50, w_2 = 0.50$)

- **TP53**: `TP53_150_-`, `TP53_11911_-`, `TP53_559_-`, `TP53_2766_+`, `TP53_3053_-`
- **HBB**: `HBB_1533_+`, `HBB_3541_+`, `HBB_1202_-`, `HBB_1861_-`, `HBB_2265_-`
- **EGFR**: `EGFR_100693_+`, `EGFR_90787_-`, `EGFR_74407_+`, `EGFR_27703_-`, `EGFR_167075_+`
- **BRCA1**: `BRCA1_2088_+`, `BRCA1_2323_+`, `BRCA1_27410_-`, `BRCA1_122416_-`, `BRCA1_74210_+`

---

## 2. Quick Start & Canonical Entry Points

### Prerequisites & Environment Setup

The project runs in Python 3.13 / 3.14 (with virtual environment `venv/`):

```bash
# Clone repository
git clone https://github.com/dakshenyaks-hub/CRISPR-GuideRNA-Optimization.git
cd CRISPR-GuideRNA-Optimization

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies (if not using pre-configured venv)
pip install -r requirements.txt
```

### 1. Launch Interactive Web Dashboard

Launch the unified 5-tab research dashboard with integrated REST API and interactive biophysical/mathematical tools:

```bash
python src/serve_dashboard.py
```
*The dashboard will automatically open at `http://localhost:8000/` (or the next available port).*

Alternative shortcut:
```bash
python src/run_dashboard.py
```

### 2. Run Frozen Baseline Integrity Self-Test

Verify that all four genes match the authoritative frozen research baseline with 100% fidelity:

```bash
python src/pipeline_controller.py
```

### 3. Run Candidate Generation & Feature Extraction CLI

Generate sgRNA candidates and compute biophysical features for any gene (e.g. TP53) using dynamic NCBI Entrez sequence retrieval:

```bash
python main.py --gene TP53
```

Options:
```bash
python main.py --help
# Run with prebuilt Bowtie index for off-target search:
python main.py --gene TP53 --bowtie-index data/hg38_index/hg38_index
```

### 4. Execute Full Automated Test Suite

Run all 95 unit, E2E, controller, and dashboard validation tests:

```bash
# Using pytest (configured via pytest.ini):
pytest

# Or using the master runner:
python tests/run_all_tests.py
```

---

## 3. Repository Architecture

```
CRISPR-GuideRNA-Optimization/
├── dashboard/                     # Web dashboard assets & data payload
│   ├── components/                # SVG Cas9 visualizer CSS & JS
│   ├── data.json                  # Canonical multi-gene JSON payload
│   └── index.html                 # 5-tab unified research application
├── data/                          # Reference sequences, indices, & CFD matrices
│   ├── cfd_reference/             # Doench et al. CFD mismatch & PAM score pickles
│   ├── hg38_index/                # Bowtie index for GRCh38 human genome
│   ├── reference_sequences/       # NCBI-retrieved genomic FASTA & metadata
│   └── ncbi_dataset(TP53)/        # NCBI datasets package archive for TP53
├── docs/                          # Architectural & technical documentation
│   └── crispr_mechanism_visualizer.md # Mechanism visualizer specification
├── models/                        # Pre-trained deep learning checkpoints
│   └── Paired-Library/            # Kim et al. DeepSpCas9 CNN PyTorch weights (.pt)
├── reports/                       # Formal research audit reports & summaries
│   ├── final_project_cleanup_report.md             # Repository cleanup audit
│   ├── dashboard_simplification_and_validation_audit.md # Navigation & data audit
│   ├── step11_sensitivity_summary.json            # Step 11 sensitivity output
│   └── step10b_optimization_summary.json          # Step 10B optimization output
├── results/                       # Canonical research datasets & optimization runs
│   ├── TP53/                      # TP53 feature tables & optimization runs
│   ├── HBB/                       # HBB feature tables & optimization runs
│   ├── EGFR/                      # EGFR feature tables & optimization runs
│   ├── BRCA1/                     # BRCA1 feature tables & optimization runs
│   ├── step10a_cfd_rescored/      # Deduplicated, verified CFD-rescored dataset
│   ├── step10b_optimization/      # Frozen baseline Pareto fronts & selections
│   └── step11_sensitivity/        # Censored-worst-case & epsilon sensitivity runs
├── scripts/                       # Research workflow & audit execution scripts
│   ├── regenerate_step10a_rescored.py # Step 10A CFD rescoring generator
│   ├── step10b_optimize.py        # Step 10B multi-objective optimization engine
│   └── step11_sensitivity.py      # Step 11 sensitivity & robustness audit
├── src/                           # Core framework source code
│   ├── cfd_score.py               # Cutting Frequency Determination model
│   ├── deep_spcas9.py             # PyTorch CNN on-target efficiency predictor
│   ├── feature_extraction.py      # Biophysical features (Tm, MFE, GC, homopolymers)
│   ├── gene_sequence_provider.py  # Sequence provider abstraction
│   ├── grna_optimizer.py          # Pareto dominance & scalarization engine
│   ├── guide_generation.py        # NGG protospacer scanning & extraction
│   ├── ncbi_sequence_provider.py  # NCBI Entrez E-utilities REST client
│   ├── pipeline_controller.py     # Decoupled data access layer & baseline guard
│   └── serve_dashboard.py         # High-performance HTTP server & REST API
├── tests/                         # Comprehensive automated test suite
│   ├── run_all_tests.py           # Master test runner
│   ├── test_cfd_score.py          # 13 CFD unit test suites
│   ├── test_cfd_e2e.py            # CFD end-to-end regression tests
│   ├── test_pipeline_controller.py # Baseline fidelity tests
│   ├── test_ncbi_sequence_provider.py # NCBI Entrez client tests
│   ├── test_optimization_lab.py   # Optimization lab math & KaTeX tests
│   ├── test_validation_dashboard.py # Navigation & audit tests
│   └── test_crispr_visualizer.py  # Visualizer DOM & sequence tests
├── .gitignore                     # Repository ignore rules (caches, venvs, OS)
├── main.py                        # Root CLI orchestrator
├── pytest.ini                     # Pytest configuration restricting to tests/
├── requirements.txt               # Pinned package requirements
└── walkthrough.md                 # NCBI integration walkthrough & history
```

---

## 4. Methodology & Theoretical Foundation

### 1. Protospacer Extraction & Feasibility Filtering
The target genomic sequence is scanned along both plus ($+$) and minus ($-$) strands for canonical SpCas9 `5'-NGG-3'` Protospacer Adjacent Motifs (PAM). Candidate 20-nt protospacers are filtered through stringent biophysical constraints:
- **GC Content**: $40\% \le \text{GC} \le 70\%$
- **Poly-T Tract Avoidance**: Maximum thymine homopolymer length $< 4$ (prevents premature Pol III U6 termination)
- **Homopolymer Penalty**: Rejection of uninterrupted runs $\ge 5$ nt
- **Duplex Thermodynamics**: Nearest-neighbor melting temperature ($T_m$) and secondary structure minimum free energy ($\text{MFE}$) computed via ViennaRNA

### 2. On-Target Activity Prediction (DeepSpCas9)
Cleavage activity ($E_i \in [0, 100]$) is modeled using a PyTorch port of the Kim et al. DeepSpCas9 convolutional neural network trained on massive high-throughput guide-target indel assays.

### 3. Genome-Wide Off-Target Risk (Bowtie + CFD)
Protospacers are mapped across the human reference genome (`GRCh38`) using Bowtie with up to 3 mismatches:
$$\text{CFD}(s, t) = \text{PAM\_Score}(t_{\text{PAM}}) \times \prod_{j=1}^{20} \text{Mismatch\_Score}(s_j, t_j, j)$$
$$\text{Off-Target Risk } R_i = \sum_{t \in \text{Off-Targets}} \text{CFD}(s_i, t)$$
$$\text{Genome-Wide Specificity } S_i = \frac{100}{100 + R_i} \in (0, 1]$$

### 4. Multi-Objective Optimization
Guides are ranked and selected across the non-dominated Pareto front balancing activity $E_i$ and specificity $S_i$:
$$\max \quad U(E_i, S_i) = w_1 \cdot E_i + w_2 \cdot (100 \cdot S_i) \quad \text{s.t.} \quad w_1 + w_2 = 1.0, \; w_1, w_2 \ge 0$$
The canonical baseline uses equal scalarization ($w_1 = 0.50, w_2 = 0.50$).

---

## 5. Scientific Disclaimer

All guide scores, on-target cleavage predictions (DeepSpCas9), and off-target specificity metrics (CFD) are computational outputs generated by machine learning models and biophysical empirical matrices. They represent mathematical optimization and do not constitute wet-lab experimental validation.
