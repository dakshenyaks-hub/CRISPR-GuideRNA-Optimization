"""
dashboard/create_bio_backend_pdf.py

Builds the comprehensive, publication-quality mathematical and biophysical backend
documentation PDF file at:
    ../data/bio backend.pdf

Covers:
  1. Complete End-to-End Mathematical Architecture & Notation
  2. Biophysical Chemistry & Thermodynamic Kinetics (SantaLucia 2004, ViennaRNA MFE)
  3. Sequence Feature Modeling & Doench 2016 On-Target Proxy
  4. Genome-Wide Alignment & Off-Target Specificity Mismatch Algebra (Bowtie 1.3.1)
  5. Multi-Objective Normalization Algebra (Min-Max, Derringer-Suich Desirability)
  6. Multi-Objective Quality Function Formulation (F_i = 0.35E + 0.40S + 0.25R)
  7. Biological Feasibility Indicator Vector & Convex Polytope Delineation
  8. Linear Programming Formulation (Primal & Dual Formulations)
  9. Convexity Proofs & Total Unimodularity (Binary Vertex Exactness)
 10. CVXOPT Standard Matrix Formulation (min q^T x s.t. Gx <= h)
 11. Karush-Kuhn-Tucker (KKT) Optimality Conditions & Primal-Dual Interior-Point Mechanics
 12. Empirical Mathematical Results on Human TP53 Dataset (2,860 -> 5 Feasible Guides)
 13. Comprehensive Mathematical Symbol & Algorithm Reference Table
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Target PDF destination in data/ directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = DATA_DIR / "bio backend.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for adding running headers and 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        if self._pageNumber > 1:
            # Header
            self.drawString(36, 11 * inch - 28, "CRISPR gRNA Optimization Framework — Mathematical & Biophysical Backend")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)

            # Footer
            footer_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * inch - 36, 25, footer_text)
            self.drawString(36, 25, "MATHEMATICAL FORMULATION & ALGORITHMIC SPECIFICATION — BIO BACKEND")
            self.line(36, 35, 8.5 * inch - 36, 35)

        self.restoreState()


def build_bio_backend_pdf():
    print(f"Building bio backend PDF at: {PDF_PATH} ...")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F172A")     # Dark Slate
    ACCENT_BLUE = colors.HexColor("#0284C7") # Azure Blue
    ACCENT_PURPLE = colors.HexColor("#7C3AED") # Deep Violet
    ACCENT_TEAL = colors.HexColor("#0D9488") # Deep Teal
    TEXT_DARK = colors.HexColor("#1E293B")   # Charcoal
    TEXT_MUTED = colors.HexColor("#64748B")  # Slate Gray
    BG_LIGHT = colors.HexColor("#F8FAFC")    # Clean Light Gray
    BG_CODE = colors.HexColor("#F1F5F9")     # Code Box Gray
    BORDER_COLOR = colors.HexColor("#CBD5E1")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=ACCENT_BLUE,
        spaceAfter=5,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=TEXT_MUTED,
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=ACCENT_PURPLE,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_DARK,
        spaceAfter=4,
    )
    body_bold = ParagraphStyle(
        "Body_Bold",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=PRIMARY,
        spaceAfter=3,
    )
    formula_style = ParagraphStyle(
        "Formula_Box",
        fontName="Courier-Bold",
        fontSize=8,
        leading=11,
        textColor=PRIMARY,
        backColor=BG_CODE,
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=5,
    )
    th_style = ParagraphStyle(
        "TH_Style",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0,
    )
    td_style = ParagraphStyle(
        "TD_Style",
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=TEXT_DARK,
        alignment=0,
    )
    td_bold = ParagraphStyle(
        "TD_Bold",
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=PRIMARY,
        alignment=0,
    )
    td_mono = ParagraphStyle(
        "TD_Mono",
        fontName="Courier",
        fontSize=6.5,
        leading=8.5,
        textColor=PRIMARY,
        alignment=0,
    )

    story = []

    # =========================================================================
    # TITLE & HEADER
    # =========================================================================
    story.append(Paragraph("CRISPR gRNA Multi-Objective Optimization Framework", title_style))
    story.append(Paragraph("Mathematical Foundations, Biophysical Backend Algorithms, &amp; Convex Optimization Architecture", subtitle_style))
    story.append(Paragraph("<b>Document Scope:</b> Comprehensive Mathematical &amp; Algorithmic Backend Specification &bull; <b>Locus:</b> Human TP53 (GRCh38.p14) &bull; <b>Population:</b> N = 2,860 Candidates &bull; <b>Solver:</b> CVXOPT Convex Linear Program", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_BLUE, spaceBefore=0, spaceAfter=8))

    # =========================================================================
    # SECTION 1: EXECUTIVE OVERVIEW & ARCHITECTURAL PIPELINE
    # =========================================================================
    story.append(Paragraph("1. Mathematical Problem Formulation &amp; Global Architecture", h1_style))
    story.append(Paragraph(
        "The multi-objective CRISPR guide RNA (gRNA) optimization framework transforms a continuous genomic locus into a globally optimal subset of guide RNAs. The computational backend is rigorously structured as a <b>Multi-Objective Constrained Linear Program (LP)</b> operating over a convex feasible polytope. Candidate protospacers are simultaneously evaluated on on-target cleavage potency (Efficiency), genome-wide non-specific cleavage risk (Specificity), and biophysical thermodynamics (Structural Suitability).",
        body_style
    ))
    story.append(Paragraph(
        "The complete transformation maps nucleotide sequences across continuous mathematical spaces as follows:",
        body_style
    ))
    story.append(Paragraph(
        "Sequence Space Sigma^20 --[Feature Extraction]--> Feature Space R^23 --[Normalization]--> Quality Domain [0, 1]^23 --[Weights]--> Objective Vector c in [0, 1]^N --[Constraints]--> Feasible Polytope F in [0, 1]^N --[CVXOPT LP]--> Optimal Solution x* in {0, 1}^N",
        formula_style
    ))

    # Architecture Table
    arch_data = [
        [Paragraph("Pipeline Stage", th_style), Paragraph("Mathematical Mapping", th_style), Paragraph("Algorithmic Transformation", th_style), Paragraph("Domain &amp; Dimensions", th_style)],
        [Paragraph("1. Protospacer Scan", td_bold), Paragraph("s_i in {A,C,G,T}^20", td_mono), Paragraph("Regular expression detection: 5'-N20-NGG-3' on forward/reverse strands", td_style), Paragraph("N = 2,860 candidates", td_style)],
        [Paragraph("2. Feature Extraction", td_bold), Paragraph("Phi: Sigma^20 -> R^23", td_mono), Paragraph("Thermodynamic NN kinetics, ViennaRNA MFE, Bowtie BWT alignment", td_style), Paragraph("23 features / guide", td_style)],
        [Paragraph("3. Normalization", td_bold), Paragraph("N: R -> [0, 1]", td_mono), Paragraph("Direction-aware Min-Max rescaling &amp; Derringer-Suich desirability", td_style), Paragraph("E_i, S_i, R_i in [0, 1]", td_style)],
        [Paragraph("4. Objective Score", td_bold), Paragraph("c_i = w^T f_norm_i", td_mono), Paragraph("Weighted linear aggregation: c_i = 0.35 E_i + 0.40 S_i + 0.25 R_i", td_style), Paragraph("c_i in [0, 1]", td_style)],
        [Paragraph("5. Feasibility Bound", td_bold), Paragraph("u_i in {0, 1}", td_mono), Paragraph("Biological indicator: GC in [40,60]%, Tm in [55,65] deg C, Off-targets &lt;= 20", td_style), Paragraph("F subset of R^2860 (5 pass)", td_style)],
        [Paragraph("6. CVXOPT Solver", td_bold), Paragraph("max c^T x s.t. x in F", td_mono), Paragraph("Primal-dual interior-point method solving bounded LP under budget K=10", td_style), Paragraph("x* = [1, 1, 1, 1, 1]^T", td_style)],
    ]
    arch_table = Table(arch_data, colWidths=[1.25 * inch, 1.35 * inch, 3.65 * inch, 1.25 * inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 2: THERMODYNAMIC & BIOPHYSICAL KINETICS
    # =========================================================================
    story.append(Paragraph("2. Thermodynamic &amp; Biophysical Chemistry Calculations", h1_style))
    story.append(Paragraph(
        "<b>A. SantaLucia &amp; Hicks (2004) Nearest-Neighbor Duplex Kinetics:</b> The thermodynamic stability of the 20-nt gRNA:DNA heteroduplex is computed via nearest-neighbor (NN) stacking interactions. Total enthalpy (dH) and entropy (dS) are computed as the sum of helix initiation, terminal A/T penalties, and 19 consecutive dinucleotide stacking steps:",
        body_style
    ))
    story.append(Paragraph(
        "dH = dH_init + dH_term(5') + dH_term(3') + sum_{j=1}^{19} dH_stack(s_j s_{j+1} / s_bar_j s_bar_{j+1})<br/>"
        "dS = dS_init + dS_term(5') + dS_term(3') + sum_{j=1}^{19} dS_stack(s_j s_{j+1} / s_bar_j s_bar_{j+1})",
        formula_style
    ))
    story.append(Paragraph(
        "The Gibbs Free Energy of Duplex Formation at physiological temperature (T = 37 deg C = 310.15 K) and the Nearest-Neighbor Melting Temperature (Tm) are determined via the standard van 't Hoff relation:",
        body_style
    ))
    story.append(Paragraph(
        "dG_37 = dH - T * dS  (kcal/mol)<br/>"
        "Tm = [ dH / ( dS + R * ln(C_T / 4) ) ] - 273.15  (deg C)<br/>"
        "where R = 1.9872 cal/(mol*K) is the universal gas constant, and C_T = 250 * 10^-9 M is the total oligonucleotide concentration.",
        formula_style
    ))
    story.append(Paragraph(
        "<b>B. ViennaRNA Secondary Structure Minimum Free Energy (MFE):</b> Self-folding of the single-stranded gRNA spacer can inhibit Cas9 ribonucleoprotein complex assembly. Secondary structure folding is modeled using dynamic programming (Zuker algorithm) via ViennaRNA. The Minimum Free Energy (dG_MFE) minimizes loop decomposition energy:",
        body_style
    ))
    story.append(Paragraph(
        "dG_MFE(s) = min_{S in ValidStructures(s)} [ sum_{h in Hairpins} dG_H(h) + sum_{i in Interior} dG_I(i) + sum_{b in Bulges} dG_B(b) + sum_{m in Multi} dG_M(m) ]<br/>"
        "Paired Base Fraction: f_paired = (1 / L) * sum_{k=1}^L I(S_k in {'(', ')'})",
        formula_style
    ))

    # =========================================================================
    # SECTION 3: SPECIFICITY MODELING & GENOME ALIGNMENT
    # =========================================================================
    story.append(Paragraph("3. Specificity Modeling, Burrows-Wheeler Transform &amp; Mismatch Algebra", h1_style))
    story.append(Paragraph(
        "<b>A. Bowtie 1.3.1 Genome-Wide Indexing:</b> All 2,860 candidate sequences are aligned against the entire human reference genome (GRCh38.p14) using the Burrows-Wheeler Transform (BWT) and FM-Index with up to 3 mismatches (-v 3 -k 1000). Exactly one 0-mismatch alignment corresponds to the intended on-target locus and is deducted from the off-target tally:",
        body_style
    ))
    story.append(Paragraph(
        "OT_0 = max(Hits_0mm - 1, 0),   OT_1 = Hits_1mm,   OT_2 = Hits_2mm,   OT_3 = Hits_3mm<br/>"
        "Total Off-Target Burden: OT_total = OT_0 + OT_1 + OT_2 + OT_3",
        formula_style
    ))
    story.append(Paragraph(
        "<b>B. Mismatch Divergence &amp; Capping Handling:</b> To account for target discrimination, the mean edit distance across off-targets is computed. Highly divergent off-targets (3 mismatches) represent lower cleavage risk than near-perfect matches (1 mismatch):",
        body_style
    ))
    story.append(Paragraph(
        "d_mean_mismatch = ( sum_{m=0}^3 m * OT_m ) / ( sum_{m=0}^3 OT_m ) = ( 0*OT_0 + 1*OT_1 + 2*OT_2 + 3*OT_3 ) / OT_total<br/>"
        "Zero Off-Target Imputation: if OT_total = 0 => d_mean_mismatch = max_j(d_mean_mismatch_j) (Safest bound)<br/>"
        "Alignment Capping Penalty: if OT_total >= 999 => OT_norm = 0.0 (Worst-burden floor for truncated searches)",
        formula_style
    ))

    # =========================================================================
    # SECTION 4: NORMALIZATION & MULTI-OBJECTIVE COMPOSITION
    # =========================================================================
    story.append(Paragraph("4. Multi-Objective Normalization &amp; Composite Scoring Algebra", h1_style))
    story.append(Paragraph(
        "Individual biological features exist across vastly different dimensional scales (deg C, kcal/mol, hit counts). To prevent scale distortion, features undergo direction-aware linear min-max normalization and non-linear desirability transforms:",
        body_style
    ))
    story.append(Paragraph(
        "Direction-Aware Linear Scaling:   N_max(X) = (X - X_min) / (X_max - X_min),    N_min(X) = 1.0 - (X - X_min) / (X_max - X_min)<br/><br/>"
        "Derringer-Suich Trapezoidal Plateau Desirability for Tm (Optimal Range [55 deg C - 65 deg C], tolerance delta = 5 deg C):<br/>"
        "d(Tm) = 1.0 if 55 &lt;= Tm &lt;= 65;   (Tm - 50)/5 if 50 &lt;= Tm &lt; 55;   (70 - Tm)/5 if 65 &lt; Tm &lt;= 70;   0.0 otherwise",
        formula_style
    ))
    story.append(Paragraph(
        "<b>Three Aggregated Sub-Objective Scores:</b>",
        body_bold
    ))
    story.append(Paragraph(
        "1. Efficiency (E_i): E_i = N_max(Doench_Proxy_i)<br/>"
        "2. Specificity (S_i): S_i = 0.70 * N_min(OT_total_i) + 0.30 * N_max(d_mean_mismatch_i)<br/>"
        "3. Structural Suitability (R_i): R_i = (1/3) * [ d(Tm_i) + N_min(dG_i) + N_max(dG_MFE_i) ]",
        formula_style
    ))
    story.append(Paragraph(
        "<b>Composite Multi-Objective Quality Function (c_i):</b>",
        body_bold
    ))
    story.append(Paragraph(
        "c_i = w_E * E_i + w_S * S_i + w_R * R_i = 0.35 * E_i + 0.40 * S_i + 0.25 * R_i    (where sum w = 1.0)",
        formula_style
    ))

    # =========================================================================
    # SECTION 5: MATHEMATICAL OPTIMIZATION, CONVEXITY & CVXOPT
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Constrained Convex Optimization, Linear Programming &amp; CVXOPT Solver", h1_style))
    story.append(Paragraph(
        "<b>A. Core Optimization Principle &amp; Distinct Mathematical Roles:</b><br/>"
        "It is mathematically crucial to distinguish the three fundamental pillars of the system:<br/>"
        "&bull; <b>Weighted Scoring Function (c_i):</b> Evaluates the multi-objective utility of candidate guides.<br/>"
        "&bull; <b>Biological Constraints (u_i):</b> Delineates the bounding hyperplanes of the feasible convex polytope F.<br/>"
        "&bull; <b>CVXOPT Linear Program:</b> Solves the resource-constrained optimization problem to find the global optimum over F.",
        body_style
    ))
    story.append(Paragraph(
        "<b>B. Biological Feasibility Indicator Vector (u in {0, 1}^N):</b><br/>"
        "A guide RNA candidate i is biologically feasible (u_i = 1) if and only if it satisfies all 4 experimental bounds simultaneously:",
        body_style
    ))
    story.append(Paragraph(
        "u_i = I(0.40 &lt;= GC_i &lt;= 0.60) * I(55.0 deg C &lt;= Tm_i &lt;= 65.0 deg C) * I(OT_total_i &lt;= 20) * I(Capped_i == False)",
        formula_style
    ))
    story.append(Paragraph(
        "<b>C. Primal Linear Program Formulation:</b>",
        body_bold
    ))
    story.append(Paragraph(
        "maximize    sum_{i=1}^N c_i * x_i  = c^T x<br/>"
        "subject to  0 &lt;= x_i &lt;= u_i    forall i in {1, ..., N}   (Box constraints bounded by feasibility)<br/>"
        "            sum_{i=1}^N x_i &lt;= K   (Selection capacity budget constraint; K = 10)",
        formula_style
    ))

    # Mathematical Proofs Box
    story.append(Paragraph("Formal Mathematical Proofs of Convexity &amp; Binary Vertex Optimality", h2_style))
    story.append(Paragraph(
        "<b>Proof 1 (Convexity of Objective Function):</b> The objective function f(x) = c^T x is a linear map from R^N to R. For any x, y in R^N and alpha in [0, 1]:<br/>"
        "    f(alpha * x + (1 - alpha) * y) = c^T(alpha * x + (1 - alpha) * y) = alpha * c^T x + (1 - alpha) * c^T y = alpha * f(x) + (1 - alpha) * f(y)<br/>"
        "Because equality holds, f(x) is both convex and concave (affine). Maximizing an affine function over a convex set is a canonical convex optimization problem.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Proof 2 (Convexity of the Feasible Set F):</b> The feasible set F is defined by 2N + 1 linear inequalities:<br/>"
        "    F = { x in R^N | -x_i &lt;= 0,  x_i &lt;= u_i,  sum_{i=1}^N x_i &lt;= K }<br/>"
        "Each linear inequality a_j^T x &lt;= b_j defines a closed halfspace H_j in R^N. Halfspaces are convex sets. The feasible region is the intersection F = intersect_{j=1}^{2N+1} H_j. Since the intersection of any collection of convex sets is convex, F is a bounded convex polytope.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Proof 3 (LP Vertex Extreme Point &amp; Binary Integrality Exactness):</b> By the Fundamental Theorem of Linear Programming, if a linear program has an optimal solution, it is attained at an extreme point (vertex) of the polytope F. The constraint matrix G is totally unimodular. Therefore, all basic feasible solutions (vertices) are integral: x_vertex in {0, 1}^N. The continuous LP relaxation solved by CVXOPT guarantees the exact combinatorial integer optimum without branch-and-bound relaxation loss.",
        body_style
    ))

    story.append(Paragraph("CVXOPT Standard Matrix Formulation (Primal-Dual Form)", h2_style))
    story.append(Paragraph(
        "CVXOPT solves convex optimization problems in standard minimization format:<br/>"
        "    <b>minimize</b>  q^T x     <b>subject to</b>  G x &lt;= h<br/>"
        "To maximize c^T x, we set q = -c in R^N. The constraint system G x &lt;= h is constructed via block matrices:",
        body_style
    ))
    story.append(Paragraph(
        "G = [ -I_{N x N} ;  I_{N x N} ;  1_{1 x N} ] in R^{(2N+1) x N}<br/>"
        "h = [ 0_{N x 1} ;  u_{N x 1} ;  K ] in R^{2N+1}",
        formula_style
    ))
    story.append(Paragraph(
        "<b>Karush-Kuhn-Tucker (KKT) Optimality Conditions:</b><br/>"
        "The primal-dual interior-point method introduces Lagrange multipliers lambda >= 0 (lower bounds), mu >= 0 (upper bounds), and gamma >= 0 (budget constraint):",
        body_style
    ))
    story.append(Paragraph(
        "1. Primal Feasibility:   -x_i &lt;= 0,  x_i - u_i &lt;= 0,  sum x_i - K &lt;= 0<br/>"
        "2. Dual Feasibility:     lambda_i >= 0,  mu_i >= 0,  gamma >= 0<br/>"
        "3. Stationarity:         grad_x L = -c - lambda + mu + gamma * 1 = 0 => c_i = mu_i - lambda_i + gamma<br/>"
        "4. Complementary Slackness: lambda_i * x_i = 0,  mu_i * (u_i - x_i) = 0,  gamma * (K - sum x_i) = 0  (forall i)",
        formula_style
    ))

    # =========================================================================
    # SECTION 6: EMPIRICAL RESULTS ON TP53 DATASET
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Empirical Dataset Results &amp; Complete Verification (Human TP53)", h1_style))
    story.append(Paragraph(
        "The multi-objective optimization framework was executed on the complete human <b>TP53</b> gene dataset (20,394 bp exonic/intronic region). A total of <b>2,860</b> unique 20-nt protospacers upstream of NGG PAM motifs were identified and evaluated genome-wide against <b>GRCh38.p14</b>.",
        body_style
    ))

    # Reduction Step Table
    red_data = [
        [Paragraph("Reduction Step", th_style), Paragraph("Filter Bound Applied", th_style), Paragraph("Surviving Candidates", th_style), Paragraph("Retention %", th_style), Paragraph("Mathematical Impact on Feasible Set", th_style)],
        [Paragraph("Step 0: Raw Candidates", td_bold), Paragraph("All NGG PAM upstream 20-nt sites", td_style), Paragraph("<b>2,860</b>", td_mono), Paragraph("100.0%", td_style), Paragraph("Unconstrained search space R^2860", td_style)],
        [Paragraph("Step 1: GC Content", td_bold), Paragraph("0.40 &lt;= GC &lt;= 0.60", td_mono), Paragraph("<b>1,708</b>", td_mono), Paragraph("59.7%", td_style), Paragraph("Eliminates unstable binding / extreme secondary folding", td_style)],
        [Paragraph("Step 2: Melting Temp (Tm)", td_bold), Paragraph("55.0 deg C &lt;= Tm &lt;= 65.0 deg C", td_mono), Paragraph("<b>991</b>", td_mono), Paragraph("34.7%", td_style), Paragraph("Restricts search to physiological annealing temperatures", td_style)],
        [Paragraph("Step 3: Off-Target Ceiling", td_bold), Paragraph("Off-Target Count &lt;= 20 hits", td_mono), Paragraph("<b>15</b>", td_mono), Paragraph("0.52%", td_style), Paragraph("Eliminates severe genome-wide promiscuous cleavage risk", td_style)],
        [Paragraph("Step 4: All Feasible Bounds", td_bold), Paragraph("GC &amp; Tm &amp; OT &lt;= 20 &amp; Uncapped", td_mono), Paragraph("<b>5</b>", td_mono), Paragraph("0.17%", td_style), Paragraph("Feasible Polytope F bounded to 5 candidate guides", td_style)],
        [Paragraph("Step 5: CVXOPT Optimal LP", td_bold), Paragraph("max c^T x s.t. x in F, sum x &lt;= 10", td_mono), Paragraph("<b>5 Selected</b>", td_mono), Paragraph("0.17%", td_style), Paragraph("Optimal vertex attainment with binary LP decisions x_i* = 1.0", td_style)],
    ]
    red_table = Table(red_data, colWidths=[1.25 * inch, 1.55 * inch, 1.05 * inch, 0.75 * inch, 2.9 * inch])
    red_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(red_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Optimal CVXOPT-Selected Guide RNAs (Complete Solution Vector)", h2_style))

    # Selected Guides Table
    sel_data = [
        [Paragraph("Rank", th_style), Paragraph("Guide Sequence (5' -> 3')", th_style), Paragraph("PAM", th_style), Paragraph("c_i", th_style), Paragraph("E_i (35%)", th_style), Paragraph("S_i (40%)", th_style), Paragraph("R_i (25%)", th_style), Paragraph("GC%", th_style), Paragraph("Tm (deg C)", th_style), Paragraph("OT", th_style), Paragraph("x_i*", th_style)],
        [Paragraph("<b>#1</b>", td_bold), Paragraph("GTACGCTTTAGGCCGGCTCA", td_mono), Paragraph("AGG", td_style), Paragraph("<b>0.8681</b>", td_bold), Paragraph("0.732", td_style), Paragraph("1.000", td_style), Paragraph("0.848", td_style), Paragraph("60.0%", td_style), Paragraph("56.72", td_style), Paragraph("9", td_style), Paragraph("<b>1.000</b>", td_bold)],
        [Paragraph("<b>#2</b>", td_bold), Paragraph("GCTGAACGTCGTGAAGCGGA", td_mono), Paragraph("AGG", td_style), Paragraph("<b>0.8554</b>", td_bold), Paragraph("0.732", td_style), Paragraph("0.998", td_style), Paragraph("0.800", td_style), Paragraph("60.0%", td_style), Paragraph("57.61", td_style), Paragraph("12", td_style), Paragraph("<b>1.000</b>", td_bold)],
        [Paragraph("<b>#3</b>", td_bold), Paragraph("ACGCTACCATGGCGTGAGAC", td_mono), Paragraph("AGG", td_style), Paragraph("<b>0.8385</b>", td_bold), Paragraph("0.732", td_style), Paragraph("0.996", td_style), Paragraph("0.736", td_style), Paragraph("60.0%", td_style), Paragraph("56.74", td_style), Paragraph("15", td_style), Paragraph("<b>1.000</b>", td_bold)],
        [Paragraph("<b>#4</b>", td_bold), Paragraph("CTGAACGTCGTGAAGCGGAA", td_mono), Paragraph("GGG", td_style), Paragraph("<b>0.7979</b>", td_bold), Paragraph("0.557", td_style), Paragraph("0.989", td_style), Paragraph("0.830", td_style), Paragraph("55.0%", td_style), Paragraph("55.21", td_style), Paragraph("15", td_style), Paragraph("<b>1.000</b>", td_bold)],
        [Paragraph("<b>#5</b>", td_bold), Paragraph("TAGGCATCACGGAGCGGTTA", td_mono), Paragraph("GGG", td_style), Paragraph("<b>0.7934</b>", td_bold), Paragraph("0.557", td_style), Paragraph("0.977", td_style), Paragraph("0.831", td_style), Paragraph("55.0%", td_style), Paragraph("55.10", td_style), Paragraph("17", td_style), Paragraph("<b>1.000</b>", td_bold)],
    ]
    sel_table = Table(sel_data, colWidths=[0.45 * inch, 1.65 * inch, 0.45 * inch, 0.65 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.55 * inch, 0.65 * inch, 0.45 * inch, 0.55 * inch])
    sel_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#7C3AED")),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(sel_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Comparison: Unconstrained Ranking vs. Constrained CVXOPT Optimization", h2_style))
    story.append(Paragraph(
        "A critical mathematical insight is why <b>unconstrained ranking fails</b>. For example, unconstrained Rank #1 (GAGGACCGTCGCAATCTGAG) achieves an unconstrained score of c_i = 0.9354. However, it exhibits a non-physiological melting temperature (Tm = 54.78 deg C &lt; 55 deg C) and excessive off-target hits (OT = 23 &gt; 20). Unconstrained ranking selects dangerous or non-functional guides, whereas CVXOPT strictly optimizes within the feasible region:",
        body_style
    ))

    comp_data = [
        [Paragraph("Guide Sequence", th_style), Paragraph("W-Rank", th_style), Paragraph("Objective c_i", th_style), Paragraph("Feasible?", th_style), Paragraph("Biological Violation Reason", th_style), Paragraph("CVXOPT LP Status", th_style)],
        [Paragraph("GAGGACCGTCGCAATCTGAG", td_mono), Paragraph("#1", td_style), Paragraph("0.9354", td_bold), Paragraph("<font color='#DC2626'><b>NO</b></font>", td_style), Paragraph("Tm = 54.78 deg C (bound [55-65]); Off-targets = 23 (ceiling &lt;= 20)", td_style), Paragraph("<font color='#DC2626'><b>Rejected (u_i = 0)</b></font>", td_style)],
        [Paragraph("GCCCGAACGCAAAGTGTCCC", td_mono), Paragraph("#2", td_style), Paragraph("0.9330", td_bold), Paragraph("<font color='#DC2626'><b>NO</b></font>", td_style), Paragraph("GC content = 65.0% (allowed [40-60%])", td_style), Paragraph("<font color='#DC2626'><b>Rejected (u_i = 0)</b></font>", td_style)],
        [Paragraph("GCAGGCTGAACGTCGTGAAG", td_mono), Paragraph("#3", td_style), Paragraph("0.9284", td_bold), Paragraph("<font color='#DC2626'><b>NO</b></font>", td_style), Paragraph("Off-target count = 38 hits (ceiling &lt;= 20)", td_style), Paragraph("<font color='#DC2626'><b>Rejected (u_i = 0)</b></font>", td_style)],
        [Paragraph("GTACGCTTTAGGCCGGCTCA", td_mono), Paragraph("#40", td_style), Paragraph("0.8681", td_bold), Paragraph("<font color='#0D9488'><b>YES</b></font>", td_style), Paragraph("Satisfies all 4 bounds (GC=60%, Tm=56.72 deg C, OT=9)", td_style), Paragraph("<font color='#7C3AED'><b>Selected (Rank #1, x_i*=1)</b></font>", td_style)],
        [Paragraph("GCTGAACGTCGTGAAGCGGA", td_mono), Paragraph("#58", td_style), Paragraph("0.8554", td_bold), Paragraph("<font color='#0D9488'><b>YES</b></font>", td_style), Paragraph("Satisfies all 4 bounds (GC=60%, Tm=57.61 deg C, OT=12)", td_style), Paragraph("<font color='#7C3AED'><b>Selected (Rank #2, x_i*=1)</b></font>", td_style)],
    ]
    comp_table = Table(comp_data, colWidths=[1.55 * inch, 0.55 * inch, 0.8 * inch, 0.65 * inch, 2.65 * inch, 1.35 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 7: MASTER MATHEMATICAL SYMBOLS & EQUATIONS REFERENCE TABLE
    # =========================================================================
    story.append(Paragraph("7. Master Mathematical Notation &amp; Algorithmic Reference", h1_style))
    sym_data = [
        [Paragraph("Symbol", th_style), Paragraph("Mathematical Concept", th_style), Paragraph("Exact Formula / Algorithmic Definition", th_style), Paragraph("Biological &amp; Optimization Role", th_style)],
        [Paragraph("s_i", td_mono), Paragraph("Guide RNA Protospacer", td_style), Paragraph("s_i in {A, C, G, T}^20", td_mono), Paragraph("20-nt target sequence upstream of 5'-NGG-3' PAM", td_style)],
        [Paragraph("dG_37", td_mono), Paragraph("Duplex Free Energy", td_style), Paragraph("dH - 310.15 * dS  (SantaLucia 2004 NN)", td_mono), Paragraph("Thermodynamic hybridization binding stability", td_style)],
        [Paragraph("Tm", td_mono), Paragraph("Melting Temperature", td_style), Paragraph("dH / (dS + R * ln(C_T/4)) - 273.15", td_mono), Paragraph("Physiological annealing threshold (optimal 55-65 deg C)", td_style)],
        [Paragraph("dG_MFE", td_mono), Paragraph("Folding Free Energy", td_style), Paragraph("min_S sum dG_loop  (ViennaRNA dynamic prog.)", td_mono), Paragraph("Penalizes stable self-folding loops / hairpins", td_style)],
        [Paragraph("OT_total", td_mono), Paragraph("Off-Target Burden", td_style), Paragraph("sum_{m=0}^3 Hits_m - 1  (Bowtie GRCh38)", td_mono), Paragraph("Total off-target cleavage count across human genome", td_style)],
        [Paragraph("d_mismatch", td_mono), Paragraph("Mismatch Divergence", td_style), Paragraph("( sum m * OT_m ) / OT_total", td_mono), Paragraph("Mean edit distance across off-target sites", td_style)],
        [Paragraph("E_i", td_mono), Paragraph("Normalized Efficiency", td_style), Paragraph("N_max(Doench_Proxy_i) in [0, 1]", td_mono), Paragraph("On-target cleavage potency proxy (35% weight)", td_style)],
        [Paragraph("S_i", td_mono), Paragraph("Normalized Specificity", td_style), Paragraph("0.70 N_min(OT_i) + 0.30 N_max(d_mismatch_i)", td_mono), Paragraph("Safety / off-target avoidance score (40% weight)", td_style)],
        [Paragraph("R_i", td_mono), Paragraph("Structural Suitability", td_style), Paragraph("(1/3) [ d(Tm_i) + N_min(dG_i) + N_max(dG_MFE_i) ]", td_mono), Paragraph("Thermodynamic stability composite (25% weight)", td_style)],
        [Paragraph("c_i", td_mono), Paragraph("Objective Coefficient", td_style), Paragraph("0.35 E_i + 0.40 S_i + 0.25 R_i", td_mono), Paragraph("Guide quality scalar utility in linear program", td_style)],
        [Paragraph("u_i", td_mono), Paragraph("Feasibility Indicator", td_style), Paragraph("I(GC in [40,60]%) * I(Tm in [55,65]) * I(OT &lt;= 20)", td_mono), Paragraph("Delineates bounding hyperplanes of polytope F", td_style)],
        [Paragraph("x_i", td_mono), Paragraph("Decision Variable", td_style), Paragraph("x_i in [0, u_i] subset of [0, 1]", td_mono), Paragraph("Continuous LP selection variable (attains binary {0,1})", td_style)],
        [Paragraph("K", td_mono), Paragraph("Selection Budget", td_style), Paragraph("sum_{i=1}^N x_i &lt;= K  (K = 10)", td_mono), Paragraph("Upper bound on total guides selected for synthesis", td_style)],
        [Paragraph("CVXOPT", td_mono), Paragraph("LP Interior Solver", td_style), Paragraph("cvxopt.solvers.lp(q, G, h)", td_mono), Paragraph("Primal-dual interior-point method finding global optimum", td_style)],
    ]
    sym_table = Table(sym_data, colWidths=[0.65 * inch, 1.35 * inch, 2.65 * inch, 2.9 * inch])
    sym_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(sym_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)")


if __name__ == "__main__":
    build_bio_backend_pdf()
