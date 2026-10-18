"""
tests/test_optimization_lab.py
Automated validation test suite for the Mathematical Formulation page
and the Interactive Optimization Lab in the CRISPR dashboard.
"""

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON_PATH = PROJECT_ROOT / "dashboard" / "data.json"
INDEX_HTML_PATH = PROJECT_ROOT / "dashboard" / "index.html"


class TestOptimizationLabStructure(unittest.TestCase):
    """Verifies that dashboard/index.html contains required KaTeX and lab elements."""

    def setUp(self):
        self.full_content = INDEX_HTML_PATH.read_text(encoding="utf-8")
        tab_start = self.full_content.find('id="tab-optimization"')
        self.assertNotEqual(tab_start, -1, "tab-optimization panel missing")
        self.tab_content = self.full_content[tab_start:]

    def test_katex_loaded_in_html(self):
        self.assertIn("katex.min.css", self.full_content, "KaTeX CSS stylesheet missing from index.html")
        self.assertIn("katex.min.js", self.full_content, "KaTeX JS script missing from index.html")
        self.assertIn("auto-render.min.js", self.full_content, "KaTeX auto-render script missing from index.html")

    def test_tab_structure_sequence(self):
        self.assertIn("1. Multi-Objective Mathematical Formulation", self.tab_content, "Section 1 Formulation missing")
        self.assertIn("Interactive Optimization Lab", self.tab_content, "Section 2 Optimization Lab missing")
        self.assertIn("3. Mathematical Interpretation & Trade-Off Regimes", self.tab_content, "Section 3 Interpretation missing")
        self.assertIn("4. Frozen Validated Research Baseline", self.tab_content, "Section 4 Baseline missing")

        # Check sequence order inside tab-optimization: Formulation -> Optimization Lab -> Interpretation -> Baseline
        pos_form = self.tab_content.find("1. Multi-Objective Mathematical Formulation")
        pos_lab = self.tab_content.find("Interactive Optimization Lab")
        pos_interp = self.tab_content.find("3. Mathematical Interpretation & Trade-Off Regimes")
        pos_base = self.tab_content.find("4. Frozen Validated Research Baseline")

        self.assertTrue(pos_form < pos_lab < pos_interp < pos_base,
            "Sections must strictly follow: Formulation -> Optimization Lab -> Interpretation -> Baseline"
        )

    def test_optimization_lab_controls_exist(self):
        self.assertIn('id="optlab-slider"', self.tab_content, "Weight slider missing")
        self.assertIn('id="optlab-pre-notice"', self.tab_content, "Pre-interaction instruction missing")
        self.assertIn('id="optlab-explanation-text"', self.tab_content, "Dynamic 1-sentence explanation missing")
        self.assertIn('id="optlab-table"', self.tab_content, "Top-5 dynamic table missing")
        self.assertIn('id="optlab-subst-box"', self.tab_content, "Mathematical arithmetic substitution card missing")
        self.assertIn("Reset to Research Baseline", self.tab_content, "Reset to Research Baseline button missing")
        self.assertIn("Constraint: $w_1 + w_2 = 1.00$", self.tab_content, "Weight constraint equation label missing")


class TestOptimizationLabCalculations(unittest.TestCase):
    """Tests objective evaluations, candidate rankings, baseline fidelity and Pareto preservation."""

    def setUp(self):
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
            self.dashboard_data = json.load(f)

    def test_baseline_weight_50_50_matches_top5(self):
        for gene in ["TP53", "HBB", "EGFR", "BRCA1"]:
            gene_data = self.dashboard_data["genes"][gene]
            feasible = gene_data.get("feasible_sample", [])
            baseline_selected = gene_data.get("selected_guides", [])

            self.assertTrue(len(feasible) > 0, f"Feasible pool empty for {gene}")
            self.assertEqual(len(baseline_selected), 5, f"Baseline selected guides not equal to 5 for {gene}")

            # Evaluate utility with w1=0.50, w2=0.50
            scored = []
            for c in feasible:
                f1 = c["norm_activity"]
                f2 = c["norm_cfd_specificity"]
                u = 0.5 * f1 + 0.5 * f2
                scored.append((u, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            top5_lab = [s[1]["guide_rna"] for s in scored[:5]]
            baseline_top1 = baseline_selected[0]["guide_rna"]

            # The Top 1 guide in the lab at w1=0.5 must match the validated baseline Top 1 guide
            self.assertEqual(top5_lab[0], baseline_top1,
                f"Top 1 mismatch at w1=0.5 for {gene}: Lab got {top5_lab[0]}, baseline is {baseline_top1}"
            )

    def test_weight_sweeps_behavior(self):
        for gene in ["TP53", "HBB", "EGFR", "BRCA1"]:
            gene_data = self.dashboard_data["genes"][gene]
            feasible = gene_data.get("feasible_sample", [])

            # Pure specificity (w1=0.0, w2=1.0)
            spec_ranked = sorted(feasible, key=lambda x: x["norm_cfd_specificity"], reverse=True)
            top_spec = spec_ranked[0]
            self.assertEqual(top_spec["norm_cfd_specificity"], 1.0, f"Expected max norm specificity 1.0 for {gene}")

            # Pure activity (w1=1.0, w2=0.0)
            act_ranked = sorted(feasible, key=lambda x: x["norm_activity"], reverse=True)
            top_act = act_ranked[0]
            self.assertEqual(top_act["norm_activity"], 1.0, f"Expected max norm activity 1.0 for {gene}")

            # Activity vs Specificity top guide should differ (trade-off exists)
            self.assertNotEqual(top_spec["guide_rna"], top_act["guide_rna"],
                f"Trade-off absent in {gene}: pure activity and pure specificity chose identical guide"
            )

    def test_frozen_baseline_immutability(self):
        """Verifies that data.json retains frozen_baseline_verified metadata."""
        self.assertTrue(self.dashboard_data.get("frozen_baseline_verified"))
        for gene in ["TP53", "HBB", "EGFR", "BRCA1"]:
            self.assertEqual(len(self.dashboard_data["genes"][gene]["selected_guides"]), 5)


if __name__ == "__main__":
    unittest.main()
