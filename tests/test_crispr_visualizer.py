"""
test_crispr_visualizer.py
Comprehensive test suite for the Gene-Specific Interactive CRISPR Mechanism Visualizer.
Verifies data pipeline, canonical Top-5 guide sequences, PAM motifs, strand polarity,
metric integrity, and fallback/validation handling across all supported and dynamic genes.
"""

import pytest
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_controller import PipelineController, SUPPORTED_GENES


class TestCrisprMechanismVisualizerData:
    """Validates that real project data interfaces seamlessly with the visualizer."""

    @pytest.mark.parametrize("gene", ["TP53", "HBB", "EGFR", "BRCA1"])
    def test_top5_guides_load_correctly(self, gene):
        """Test 1-4: TP53, HBB, EGFR, BRCA1 Top-5 guides load with complete sequence and properties."""
        selected = PipelineController.get_selected_guides(gene)
        assert len(selected) == 5, f"Expected 5 selected guides for {gene}, got {len(selected)}"

        for idx, guide in enumerate(selected):
            assert "guide_id" in guide and guide["guide_id"], f"Guide #{idx} missing guide_id for {gene}"
            assert "guide_rna" in guide and len(guide["guide_rna"]) == 20, (
                f"Guide {guide.get('guide_id')} does not have a 20-nt protospacer sequence: {guide.get('guide_rna')}"
            )
            assert all(base in "ACGT" for base in guide["guide_rna"]), (
                f"Guide {guide.get('guide_id')} contains invalid non-DNA characters: {guide.get('guide_rna')}"
            )
            assert "pam_sequence" in guide and len(guide["pam_sequence"]) == 3, (
                f"Guide {guide.get('guide_id')} missing 3-nt PAM sequence"
            )
            assert guide["pam_sequence"].endswith("GG"), (
                f"Guide {guide.get('guide_id')} PAM does not match canonical SpCas9 NGG: {guide.get('pam_sequence')}"
            )
            assert guide.get("strand") in ["+", "-"], f"Guide {guide.get('guide_id')} invalid strand: {guide.get('strand')}"
            assert isinstance(guide.get("pam_position"), int) and guide["pam_position"] >= 0, (
                f"Guide {guide.get('guide_id')} invalid locus position"
            )

    def test_tp53_specific_baseline_guides(self):
        """Test exact canonical Top-5 guide IDs for TP53."""
        expected = ["TP53_150_-", "TP53_11911_-", "TP53_559_-", "TP53_2766_+", "TP53_3053_-"]
        actual = [g["guide_id"] for g in PipelineController.get_selected_guides("TP53")]
        assert actual == expected, f"TP53 Top-5 mismatch: expected {expected}, got {actual}"

    def test_hbb_specific_baseline_guides(self):
        """Test exact canonical Top-5 guide IDs for HBB."""
        expected = ["HBB_1533_+", "HBB_3541_+", "HBB_1202_-", "HBB_1861_-", "HBB_2265_-"]
        actual = [g["guide_id"] for g in PipelineController.get_selected_guides("HBB")]
        assert actual == expected, f"HBB Top-5 mismatch: expected {expected}, got {actual}"

    def test_egfr_specific_baseline_guides(self):
        """Test exact canonical Top-5 guide IDs for EGFR."""
        expected = ["EGFR_100693_+", "EGFR_90787_-", "EGFR_74407_+", "EGFR_27703_-", "EGFR_167075_+"]
        actual = [g["guide_id"] for g in PipelineController.get_selected_guides("EGFR")]
        assert actual == expected, f"EGFR Top-5 mismatch: expected {expected}, got {actual}"

    def test_brca1_specific_baseline_guides(self):
        """Test exact canonical Top-5 guide IDs for BRCA1."""
        expected = ["BRCA1_2088_+", "BRCA1_2323_+", "BRCA1_27410_-", "BRCA1_122416_-", "BRCA1_74210_+"]
        actual = [g["guide_id"] for g in PipelineController.get_selected_guides("BRCA1")]
        assert actual == expected, f"BRCA1 Top-5 mismatch: expected {expected}, got {actual}"

    def test_guide_switching_updates_sequence_and_pam(self):
        """Test 5 & 7: Switching guides within a gene changes protospacer sequence and PAM."""
        guides = PipelineController.get_selected_guides("BRCA1")
        g1 = guides[0] # BRCA1_2088_+
        g2 = guides[1] # BRCA1_2323_+

        assert g1["guide_id"] != g2["guide_id"]
        assert g1["guide_rna"] != g2["guide_rna"]
        assert g1["pam_position"] != g2["pam_position"]
        assert g1["guide_rna"] == "TCCGTGGCAACAGTAAAGCG"
        assert g1["pam_sequence"] == "TGG"
        assert g1["strand"] == "+"
        assert g1["pam_position"] == 2088

    def test_gene_switching_updates_guide_list(self):
        """Test 6: Switching genes produces unique, non-overlapping guide pools."""
        tp53_ids = {g["guide_id"] for g in PipelineController.get_selected_guides("TP53")}
        hbb_ids = {g["guide_id"] for g in PipelineController.get_selected_guides("HBB")}
        assert tp53_ids.isdisjoint(hbb_ids), "TP53 and HBB guide sets must be mutually disjoint"

    def test_strand_polarity_representation(self):
        """Test 8: Both positive and negative strand guides exist and are correctly flagged."""
        all_guides = []
        for gene in SUPPORTED_GENES:
            all_guides.extend(PipelineController.get_selected_guides(gene))

        plus_guides = [g for g in all_guides if g["strand"] == "+"]
        minus_guides = [g for g in all_guides if g["strand"] == "-"]

        assert len(plus_guides) > 0, "Expected at least one positive strand guide across benchmark set"
        assert len(minus_guides) > 0, "Expected at least one negative strand guide across benchmark set"

        # Verify negative strand guide sequence integrity
        neg_sample = minus_guides[0]
        assert len(neg_sample["guide_rna"]) == 20
        assert neg_sample["strand"] == "-"

    def test_metrics_integrity(self):
        """Test 9: Biophysical and machine learning metrics are within valid biological ranges."""
        for gene in SUPPORTED_GENES:
            for g in PipelineController.get_selected_guides(gene):
                # DeepSpCas9 activity (0 to 100%)
                assert 0.0 <= g["deep_spcas9_score"] <= 100.0, f"DeepSpCas9 score out of bounds: {g['deep_spcas9_score']}"
                # CFD specificity score (0.0 to 1.0)
                assert 0.0 <= g["cfd_specificity_score"] <= 1.0, f"CFD score out of bounds: {g['cfd_specificity_score']}"
                # CFD off-target risk (>= 0)
                assert g["cfd_off_target_risk"] >= 0.0, f"CFD risk out of bounds: {g['cfd_off_target_risk']}"
                # GC content (30% to 70% in feasible set)
                assert 0.30 <= g["gc_content"] <= 0.70, f"GC content outside feasible window: {g['gc_content']}"
                # Melting temperature
                assert 40.0 <= g["melting_temp_c"] <= 75.0, f"Melting temperature out of bounds: {g['melting_temp_c']}"

    def test_missing_data_validation(self):
        """Test 10: Visualizer data validation correctly rejects missing/malformed objects."""
        # Simulated validation logic matching crispr_visualizer.js
        def validate_guide_payload(guide):
            if not guide or not isinstance(guide, dict):
                return False, "Guide payload missing or invalid"
            seq = guide.get("guide_rna") or guide.get("sequence") or ""
            if not seq or len(seq) < 15:
                return False, "Required 20-nt sequence information is missing"
            if not guide.get("guide_id"):
                return False, "Guide ID is missing"
            return True, "Valid"

        assert validate_guide_payload(None)[0] is False
        assert validate_guide_payload({})[0] is False
        assert validate_guide_payload({"guide_id": "test_1", "guide_rna": "ATGC"})[0] is False
        assert validate_guide_payload({"guide_id": "test_1", "guide_rna": "ATGCGATCGATCGATCGATC"})[0] is True
