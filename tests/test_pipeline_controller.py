"""
tests/test_pipeline_controller.py

Unit and regression tests for src/pipeline_controller.py against
the frozen Step 10B/11 research baseline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_controller import PipelineController, FROZEN_BASELINE, SUPPORTED_GENES


def test_supported_genes():
    genes = PipelineController.get_supported_genes()
    assert set(genes) == {"TP53", "HBB", "EGFR", "BRCA1"}
    print("  [PASS] test_supported_genes")


def test_frozen_baseline_integrity():
    report = PipelineController.verify_frozen_baseline()
    assert report["all_passed"] is True, f"Baseline mismatch: {report}"
    print("  [PASS] test_frozen_baseline_integrity (All 4 genes matched)")


def test_gene_overview_metrics():
    for gene, expected in FROZEN_BASELINE.items():
        ov = PipelineController.get_gene_overview(gene)
        assert ov["total_candidates"] == expected["total_candidates"], (
            f"[{gene}] Total candidates mismatch: {ov['total_candidates']} != {expected['total_candidates']}"
        )
        assert ov["feasible_candidates"] == expected["feasible_candidates"], (
            f"[{gene}] Feasible mismatch: {ov['feasible_candidates']} != {expected['feasible_candidates']}"
        )
        assert ov["pareto_front_size"] == expected["pareto_size"], (
            f"[{gene}] Pareto size mismatch: {ov['pareto_front_size']} != {expected['pareto_size']}"
        )
        assert ov["selected_guide_ids"] == expected["selected_top5"], (
            f"[{gene}] Top-5 IDs mismatch: {ov['selected_guide_ids']} != {expected['selected_top5']}"
        )
        # Check exclusion math
        assert ov["total_candidates"] == ov["feasible_candidates"] + ov["excluded_candidates"]
        print(f"  [PASS] test_gene_overview_metrics for {gene}")


def test_pareto_front_records():
    for gene in SUPPORTED_GENES:
        pareto = PipelineController.get_pareto_front(gene)
        assert len(pareto) == FROZEN_BASELINE[gene]["pareto_size"]
        for r in pareto:
            assert "guide_id" in r
            assert "norm_activity" in r
            assert "norm_cfd_specificity" in r
            assert 0.0 <= r["norm_activity"] <= 1.000001
            assert 0.0 <= r["norm_cfd_specificity"] <= 1.000001
        print(f"  [PASS] test_pareto_front_records for {gene}")


def test_selected_guides_records():
    for gene in SUPPORTED_GENES:
        sel = PipelineController.get_selected_guides(gene)
        assert len(sel) == 5
        sel_ids = [r["guide_id"] for r in sel]
        assert sel_ids == FROZEN_BASELINE[gene]["selected_top5"]
        for r in sel:
            assert "guide_rna" in r
            assert len(r["guide_rna"]) == 20
            assert "deep_spcas9_score" in r
            assert "cfd_specificity_score" in r
            assert "cfd_off_target_risk" in r
        print(f"  [PASS] test_selected_guides_records for {gene}")


def test_sensitivity_summary():
    for gene in SUPPORTED_GENES:
        sens = PipelineController.get_sensitivity_summary(gene)
        cw = sens["censored_worst_case"]
        eps = sens["epsilon_resolution"]
        assert cw["is_100pct_stable"] is True
        assert cw["overlap_with_primary_pct"] == 100.0
        assert eps["resolution_stable"] is True
        assert eps["pareto_size_delta"] == 0
        print(f"  [PASS] test_sensitivity_summary for {gene}")


def test_full_payload_generation():
    payload = PipelineController.get_full_dashboard_payload()
    assert payload["supported_genes"] == SUPPORTED_GENES
    assert len(payload["master_comparison"]) == 4
    for g in SUPPORTED_GENES:
        assert g in payload["genes"]
        gdata = payload["genes"][g]
        assert "overview" in gdata
        assert "pareto_front" in gdata
        assert "selected_guides" in gdata
        assert "feasible_sample" in gdata
        assert "sensitivity" in gdata
    print("  [PASS] test_full_payload_generation")


def main():
    print("=" * 64)
    print("RUNNING PIPELINE CONTROLLER TESTS")
    print("=" * 64)
    test_supported_genes()
    test_frozen_baseline_integrity()
    test_gene_overview_metrics()
    test_pareto_front_records()
    test_selected_guides_records()
    test_sensitivity_summary()
    test_full_payload_generation()
    print("=" * 64)
    print("ALL PIPELINE CONTROLLER TESTS PASSED (100% BASELINE FIDELITY)")
    print("=" * 64)


if __name__ == "__main__":
    main()
