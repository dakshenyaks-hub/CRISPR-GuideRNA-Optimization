"""
tests/test_validation_dashboard.py

Automated test suite verifying the simplified dashboard navigation,
preservation of biophysical/methodological content, and the combined
Validation & Robustness page metrics and data integrity.
"""

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML_PATH = PROJECT_ROOT / "dashboard" / "index.html"
DATA_JSON_PATH = PROJECT_ROOT / "dashboard" / "data.json"


class TestDashboardNavigationAndStructure(unittest.TestCase):
    """Verifies navigation simplification and content relocation integrity."""

    def setUp(self):
        self.html = INDEX_HTML_PATH.read_text(encoding="utf-8")

    def test_navigation_has_exact_five_tabs(self):
        """Verify navigation contains exactly 5 tabs and removes Biology/Interpretation tabs."""
        nav_match = re.search(r'<nav class="main-nav">(.*?)</nav>', self.html, re.DOTALL)
        self.assertIsNotNone(nav_match, "main-nav element missing")
        nav_content = nav_match.group(1)

        # Tab button checks
        buttons = re.findall(r'<button class="nav-tab-btn[^"]*" onclick="switchNavTab\(\'([^\']+)\'\)">', nav_content)
        self.assertEqual(len(buttons), 5, f"Expected exactly 5 nav buttons, got {len(buttons)}: {buttons}")
        self.assertEqual(buttons, ['lab', 'mechanism', 'analysis', 'optimization', 'validation'])

        # Verify old separate tabs are removed from main navigation
        self.assertNotIn("switchNavTab('biology')", nav_content)
        self.assertNotIn("switchNavTab('interpretation')", nav_content)
        self.assertNotIn("switchNavTab('multigene')", nav_content)
        self.assertNotIn("switchNavTab('robustness')", nav_content)

    def test_crispr_mechanism_preserves_biology_foundations(self):
        """Verify tab-mechanism contains both visualizer and biophysical foundations."""
        mech_panel = re.search(r'<div id="tab-mechanism" class="tab-panel">(.*?)(?=<div id="tab-|</main>)', self.html, re.DOTALL)
        self.assertIsNotNone(mech_panel, "tab-mechanism panel missing")
        content = mech_panel.group(1)

        # Visualizer root must be present
        self.assertIn('id="crispr-mechanism-root"', content)

        # Biological foundations sections preserved
        self.assertIn("CRISPR/Cas9 Biophysical Foundations", content)
        self.assertIn("1. The Cas9 Endonuclease & sgRNA Anatomy", content)
        self.assertIn("2. Protospacer Adjacent Motif (PAM) Recognition", content)
        self.assertIn("3. On-Target Activity & DeepSpCas9", content)
        self.assertIn("4. Genome-Wide Off-Target Mutagenesis", content)
        self.assertIn("5. Cutting Frequency Determination (CFD) Model", content)
        self.assertIn("6. Biological Feasibility Constraints", content)

    def test_optimization_tab_preserves_interpretation_faqs(self):
        """Verify tab-optimization contains Section 5 FAQs relocated from Interpretation."""
        opt_panel = re.search(r'<div id="tab-optimization" class="tab-panel">(.*?)(?=<div id="tab-|</main>)', self.html, re.DOTALL)
        self.assertIsNotNone(opt_panel, "tab-optimization panel missing")
        content = opt_panel.group(1)

        self.assertIn("5. Methodological Interpretation & Practical Guidance", content)
        self.assertIn("5.1 What does a high DeepSpCas9 score mean?", content)
        self.assertIn("5.2 What does high CFD specificity mean?", content)
        self.assertIn("5.3 Why are some guides removed in the Feasibility Filter?", content)
        self.assertIn("5.4 Why are there multiple Pareto-optimal guides?", content)
        self.assertIn("5.5 Why did we select five guides for the portfolio?", content)

    def test_validation_robustness_panel_structure(self):
        """Verify tab-validation contains Section A, Section B, and Section C."""
        val_panel = re.search(r'<div id="tab-validation" class="tab-panel">(.*?)(?=<div id="tab-|</main>)', self.html, re.DOTALL)
        self.assertIsNotNone(val_panel, "tab-validation panel missing")
        content = val_panel.group(1)

        # Header and scope
        self.assertIn("Validation & Robustness Framework", content)
        self.assertIn("Scientific Scope & Scope Boundaries", content)
        self.assertIn("does not prove universal applicability", content)

        # Section A: Multi-Gene Benchmark
        self.assertIn("Section A — Multi-Gene Validation Benchmark", content)
        self.assertIn('id="multigene-tbody"', content)

        # Section B: Robustness & Sensitivity
        self.assertIn("Section B — Robustness & Sensitivity Analysis", content)
        self.assertIn('id="robustness-tbody"', content)
        self.assertIn("B.1 Censored Off-Target Imputation Sensitivity", content)
        self.assertIn("B.2 Epsilon-Constraint Discretization Invariance", content)
        self.assertIn("B.3 Rank Correlation & Objective Monotonicity Audit", content)

        # Section C: Research Limitations
        self.assertIn("SECTION C: RESEARCH LIMITATIONS & SCIENTIFIC NOTICE", content)
        self.assertIn("In Silico Cleavage Predictions", content)
        self.assertIn("Empirical CFD Matrix Bounds", content)
        self.assertIn("Reference Genome Assembly", content)

        # Status badge classes
        self.assertIn("status-badge-verified", self.html)
        self.assertIn("status-badge-changed", self.html)


class TestValidationDataIntegrity(unittest.TestCase):
    """Verifies that all displayed metrics match verified canonical disk data."""

    def setUp(self):
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_four_genes_authoritative_baseline(self):
        """Verify the 4-gene canonical counts: 42,733 total, 23,881 feasible, 36 Pareto, 20 selected."""
        mc = self.data["master_comparison"]
        self.assertEqual(len(mc), 4)

        by_gene = {row["gene"]: row for row in mc}
        self.assertEqual(by_gene["TP53"]["total_candidates"], 3554)
        self.assertEqual(by_gene["TP53"]["feasible_candidates"], 1648)
        self.assertEqual(by_gene["TP53"]["pareto_size"], 8)
        self.assertEqual(by_gene["TP53"]["selected_count"], 5)

        self.assertEqual(by_gene["HBB"]["total_candidates"], 540)
        self.assertEqual(by_gene["HBB"]["feasible_candidates"], 365)
        self.assertEqual(by_gene["HBB"]["pareto_size"], 10)
        self.assertEqual(by_gene["HBB"]["selected_count"], 5)

        self.assertEqual(by_gene["EGFR"]["total_candidates"], 23112)
        self.assertEqual(by_gene["EGFR"]["feasible_candidates"], 16090)
        self.assertEqual(by_gene["EGFR"]["pareto_size"], 8)
        self.assertEqual(by_gene["EGFR"]["selected_count"], 5)

        self.assertEqual(by_gene["BRCA1"]["total_candidates"], 15527)
        self.assertEqual(by_gene["BRCA1"]["feasible_candidates"], 5778)
        self.assertEqual(by_gene["BRCA1"]["pareto_size"], 10)
        self.assertEqual(by_gene["BRCA1"]["selected_count"], 5)

        total_cands = sum(row["total_candidates"] for row in mc)
        total_feas = sum(row["feasible_candidates"] for row in mc)
        total_pareto = sum(row["pareto_size"] for row in mc)
        total_selected = sum(row["selected_count"] for row in mc)

        self.assertEqual(total_cands, 42733, "Total candidates across 4 genes must equal 42,733")
        self.assertEqual(total_feas, 23881, "Total feasible candidates across 4 genes must equal 23,881")
        self.assertEqual(total_pareto, 36, "Total Pareto front size across 4 genes must equal 36")
        self.assertEqual(total_selected, 20, "Total selected guides across 4 genes must equal 20")

    def test_brca1_pareto_shift_documented(self):
        """Verify BRCA1 Pareto shift under imputation (10 to 11) is captured in HTML."""
        html = INDEX_HTML_PATH.read_text(encoding="utf-8")
        self.assertIn("CHANGED (Pareto 10&rarr;11)", html)
        self.assertIn("In BRCA1, the Pareto front expands from 10 to 11 guides", html)


if __name__ == "__main__":
    unittest.main()
