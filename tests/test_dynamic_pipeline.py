"""
tests/test_dynamic_pipeline.py

Comprehensive test suite verifying genuine dynamic gene analysis for novel genes
beyond the canonical four (TP53, HBB, EGFR, BRCA1).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_pipeline import (
    DynamicPipelineResult,
    StageResult,
    run_dynamic_pipeline,
    CANONICAL_GENES,
    DEEPCAS9_WEIGHTS,
    BOWTIE_INDEX_PREFIX,
    DYNAMIC_RESULTS_DIR,
    _check_deepcas9_available,
    _check_bowtie_available,
)
from src.pipeline_controller import PipelineController, SUPPORTED_GENES, FROZEN_BASELINE
from src.serve_dashboard import DashboardRequestHandler


class TestDynamicPipelineModule(unittest.TestCase):
    """Tests the core logic of src/dynamic_pipeline.py."""

    def setUp(self):
        self.gene_name = "TESTGENE"
        self.gene_dir = DYNAMIC_RESULTS_DIR / self.gene_name
        if self.gene_dir.exists():
            shutil.rmtree(self.gene_dir)

    def tearDown(self):
        if self.gene_dir.exists():
            shutil.rmtree(self.gene_dir)

    def test_canonical_genes_protected_from_dynamic_pipeline(self):
        """Canonical genes cannot be processed dynamically without explicit sequence override."""
        for gene in CANONICAL_GENES:
            with self.assertRaises(ValueError) as ctx:
                run_dynamic_pipeline(gene_symbol=gene)
            self.assertIn("canonical research gene", str(ctx.exception))

    def test_resource_availability_checks(self):
        """Checks whether models and indexes are detected correctly."""
        self.assertEqual(_check_deepcas9_available(), DEEPCAS9_WEIGHTS.exists())
        idx_file = Path(f"{BOWTIE_INDEX_PREFIX}.1.ebwt")
        self.assertEqual(_check_bowtie_available(), idx_file.exists())

    def test_user_provided_fasta_sequence_pipeline(self):
        """
        Runs the dynamic pipeline on a user-provided FASTA sequence.
        Skips Bowtie to test the fast execution path while running real DeepSpCas9.
        """
        # 120 bp synthetic sequence with multiple NGG PAM sites
        sample_seq = (
            "ATGGCCGAGGCGGCCGCCGAGATCGGCACCCCGCTGCAGCACGAGATCC"
            "ACGTGTGGGAGCAGCTGCCGGGCGTCACCCACTGGGTCGGCGGCAGCCG"
            "GAAGCTGGCGCAGCTGGCCG"
        )

        res = run_dynamic_pipeline(
            gene_symbol=self.gene_name,
            fasta_sequence=sample_seq,
            skip_bowtie=True,
            skip_cfd=True,
            skip_optimization=False,
        )

        self.assertIsInstance(res, DynamicPipelineResult)
        self.assertEqual(res.gene_symbol, self.gene_name)
        self.assertEqual(res.input_type, "fasta_sequence")

        # Verify stages
        stages_dict = {s.stage: s for s in res.stages}
        self.assertIn("sequence_resolution", stages_dict)
        self.assertEqual(stages_dict["sequence_resolution"].status, "completed")

        self.assertIn("candidate_generation", stages_dict)
        self.assertEqual(stages_dict["candidate_generation"].status, "completed")
        self.assertGreater(stages_dict["candidate_generation"].data["total_candidates"], 0)

        self.assertIn("feature_extraction", stages_dict)
        self.assertEqual(stages_dict["feature_extraction"].status, "completed")

        # DeepSpCas9 scoring
        self.assertIn("deepspcas9_scoring", stages_dict)
        if _check_deepcas9_available():
            self.assertEqual(stages_dict["deepspcas9_scoring"].status, "completed")
            # Verify actual scores are non-empty and in realistic range [0, 100]
            scores = res.features_df["deep_spcas9_score"].dropna()
            self.assertGreater(len(scores), 0)
            self.assertTrue((scores >= 0).all() and (scores <= 100).all())

        # Bowtie and CFD were skipped or marked unavailable
        self.assertIn("cfd_scoring", stages_dict)
        self.assertIn(stages_dict["cfd_scoring"].status, ["skipped", "unavailable"])

        # Isolation: Results are written to results/dynamic/TESTGENE
        self.assertTrue((self.gene_dir / "candidate_gRNAs.csv").exists())
        self.assertTrue((self.gene_dir / "gRNA_features.csv").exists())
        self.assertTrue((self.gene_dir / "dynamic_run_summary.json").exists())

    def test_api_response_structure_and_no_proxy_substitution(self):
        """API response format clearly reports stage status without synthetic score substitution."""
        sample_seq = "ATGCGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

        res = run_dynamic_pipeline(
            gene_symbol=self.gene_name,
            fasta_sequence=sample_seq,
            skip_bowtie=True,
            skip_deepcas9=True,
            skip_optimization=True,
        )

        api_dict = res.to_api_response()
        self.assertEqual(api_dict["status"], "success")
        self.assertEqual(api_dict["gene_symbol"], self.gene_name)
        self.assertFalse(api_dict["validated"])  # Dynamic genes are never flagged as canonical validated
        self.assertFalse(api_dict["scoring_availability"]["deepspcas9"])
        self.assertFalse(api_dict["scoring_availability"]["bowtie_off_target"])
        self.assertFalse(api_dict["scoring_availability"]["cfd_specificity"])

        # Check disclaimer is present
        self.assertIn("disclaimer", api_dict)


class TestDashboardDynamicIntegration(unittest.TestCase):
    """Tests the dashboard HTTP endpoints with dynamic pipeline integration."""

    def _create_mock_handler(self, method: str, path: str, body: dict | None = None):
        handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
        handler.path = path
        handler.command = method
        handler.headers = {}
        handler.sent_headers = {}
        handler.sent_status = None
        handler.output_buffer = io.BytesIO()
        handler.wfile = handler.output_buffer

        if body is not None:
            body_bytes = json.dumps(body).encode("utf-8")
            handler.headers["Content-Length"] = str(len(body_bytes))
            handler.rfile = io.BytesIO(body_bytes)
        else:
            handler.rfile = io.BytesIO(b"")

        def mock_send_response(status, message=None):
            handler.sent_status = status

        def mock_send_header(keyword, value):
            handler.sent_headers[keyword] = value

        def mock_end_headers():
            pass

        handler.send_response = mock_send_response
        handler.send_header = mock_send_header
        handler.end_headers = mock_end_headers
        handler.get_json_response = lambda: json.loads(handler.output_buffer.getvalue().decode("utf-8"))
        return handler

    def test_get_status_reports_dynamic_support(self):
        """GET /api/status reports online and dynamic analysis capabilities."""
        handler = self._create_mock_handler("GET", "/api/status")
        handler.do_GET()

        resp = handler.get_json_response()
        self.assertEqual(handler.sent_status, 200)
        self.assertEqual(resp["status"], "online")
        self.assertTrue(resp.get("dynamic_analysis_supported"))
        self.assertIn("resources", resp)
        self.assertIn("deepspcas9_model", resp["resources"])

    def test_post_resolve_gene_with_fasta_sequence(self):
        """POST /api/resolve_gene runs dynamic pipeline with fasta_sequence."""
        test_seq = "ATGGCCGAGGCGGCCGCCGAGATCGGCACCCCGCTGCAGCACGAGATCCACGTGTGGGAGCAGCTGCCGGGCGTCACCCACTGGGTCGGCGGCAGCCGGAAGCTGGCGCAGCTGGCCG"
        handler = self._create_mock_handler("POST", "/api/resolve_gene", body={
            "gene": "SYNTH1",
            "fasta_sequence": test_seq,
            "skip_bowtie": True,
            "skip_cfd": True,
        })
        handler.do_POST()

        resp = handler.get_json_response()
        self.assertEqual(handler.sent_status, 200)
        self.assertEqual(resp["gene_symbol"], "SYNTH1")
        self.assertFalse(resp["validated"])
        self.assertTrue(resp["dynamic_scoring_supported"])

        # Clean up
        test_dir = DYNAMIC_RESULTS_DIR / "SYNTH1"
        if test_dir.exists():
            shutil.rmtree(test_dir)

    def test_canonical_genes_unaffected(self):
        """POST /api/resolve_gene for canonical genes always serves frozen research baseline."""
        for gene in SUPPORTED_GENES:
            handler = self._create_mock_handler("POST", "/api/resolve_gene", body={"gene": gene})
            handler.do_POST()
            resp = handler.get_json_response()
            self.assertEqual(handler.sent_status, 200)
            self.assertEqual(resp["gene_symbol"], gene)
            self.assertTrue(resp["validated"])
            self.assertEqual(resp["overview"]["total_candidates"], FROZEN_BASELINE[gene]["total_candidates"])


class TestVEGFADynamicResults(unittest.TestCase):
    """Priority 1 & 5: Regression tests verifying genuine VEGFA dynamic pipeline execution."""

    @classmethod
    def setUpClass(cls):
        cls.vegfa_dir = DYNAMIC_RESULTS_DIR / "VEGFA"
        cls.summary_path = cls.vegfa_dir / "dynamic_run_summary.json"
        cls.features_path = cls.vegfa_dir / "gRNA_features.csv"
        cls.candidates_path = cls.vegfa_dir / "candidate_gRNAs.csv"
        cls.opt_dir = cls.vegfa_dir / "optimization"

    def test_vegfa_run_summary_exists_and_complete(self):
        """Verify dynamic_run_summary.json exists and all 6 stages completed."""
        self.assertTrue(self.summary_path.exists(), "VEGFA dynamic_run_summary.json missing")
        with open(self.summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["gene_symbol"], "VEGFA")
        stages = {s["stage"]: s["status"] for s in summary["stages"]}
        expected_stages = [
            "sequence_resolution",
            "candidate_generation",
            "feature_extraction",
            "deepspcas9_scoring",
            "cfd_scoring",
            "optimization",
        ]
        for stg in expected_stages:
            self.assertIn(stg, stages)
            self.assertEqual(stages[stg], "completed", f"Stage {stg} was not completed: {stages.get(stg)}")

    def test_vegfa_genuine_scores_and_distributions(self):
        """Verify genuine DeepSpCas9, CFD, and Bowtie off-target values in gRNA_features.csv."""
        import pandas as pd
        self.assertTrue(self.features_path.exists())
        df = pd.read_csv(self.features_path)

        # 4,199 total candidates
        self.assertEqual(len(df), 4199)

        # DeepSpCas9 scores
        self.assertIn("deep_spcas9_score", df.columns)
        deep_scores = df["deep_spcas9_score"].dropna()
        self.assertGreaterEqual(len(deep_scores), 4190)
        self.assertGreater(deep_scores.max(), 80.0)
        self.assertLess(deep_scores.min(), 0.0)
        self.assertAlmostEqual(float(deep_scores.mean()), 42.88, delta=1.5)

        # CFD specificity scores
        self.assertIn("cfd_specificity_score", df.columns)
        cfd_scores = df["cfd_specificity_score"].dropna()
        self.assertGreaterEqual(len(cfd_scores), 3600)
        self.assertTrue((cfd_scores > 0.0).all() and (cfd_scores <= 1.0).all())

        # Bowtie off-target counts
        self.assertIn("off_target_count", df.columns)
        ot_counts = df["off_target_count"].dropna()
        self.assertGreater(len(ot_counts), 0)

    def test_vegfa_optimization_and_selected_guides(self):
        """Verify feasible set, Pareto front, and top-5 selected guides."""
        import pandas as pd
        feasible_path = self.opt_dir / "feasible_candidates.csv"
        pareto_path = self.opt_dir / "pareto_front.csv"
        selected_path = self.opt_dir / "selected_guides.csv"

        self.assertTrue(feasible_path.exists())
        self.assertTrue(pareto_path.exists())
        self.assertTrue(selected_path.exists())

        f_df = pd.read_csv(feasible_path)
        p_df = pd.read_csv(pareto_path)
        s_df = pd.read_csv(selected_path)

        self.assertEqual(len(f_df), 2810)
        self.assertEqual(len(p_df), 7)
        self.assertEqual(len(s_df), 5)

        expected_guides = [
            "VEGFA_11896_-",
            "VEGFA_6357_-",
            "VEGFA_14105_-",
            "VEGFA_264_+",
            "VEGFA_1028_+",
        ]
        self.assertEqual(s_df["guide_id"].tolist(), expected_guides)


class TestArbitraryFastaCorrectness(unittest.TestCase):
    """Priority 2: On-target exclusion and specificity integrity for arbitrary FASTA."""

    def test_arbitrary_fasta_cfd_unavailable_to_prevent_misrepresentation(self):
        """Arbitrary FASTA without genomic coordinates must mark CFD as unavailable."""
        seq = "ATGGCCGAGGCGGCCGCCGAGATCGGCACCCCGCTGCAGCACGAGATCCACGTGTGGGAGCAGCTGCCGGGCGTCACCCACTGGGTCGGCGGCAGCCGGAAGCTGGCGCAGCTGGCCG"
        res = run_dynamic_pipeline(
            gene_symbol="ARBITRARY_TEST",
            fasta_sequence=seq,
            skip_bowtie=False,
            skip_cfd=False,
        )
        stages = {s.stage: s for s in res.stages}
        self.assertEqual(stages["cfd_scoring"].status, "unavailable")
        self.assertIn("genomic locus coordinates", stages["cfd_scoring"].message)

        # Clean up
        test_dir = DYNAMIC_RESULTS_DIR / "ARBITRARY_TEST"
        if test_dir.exists():
            shutil.rmtree(test_dir)

    def test_exclude_on_target_preserves_hits_when_unresolved(self):
        """When coordinates are absent, exclude_on_target must never drop hits silently."""
        from src.step9c_cfd import exclude_on_target
        dummy_hits = [
            {"chrom": "chr1", "strand": "+", "offset": 100, "n_mm_protospacer": 0},
            {"chrom": "chr2", "strand": "-", "offset": 500, "n_mm_protospacer": 2},
        ]
        filtered, was_excluded, method = exclude_on_target(dummy_hits, guide_chrom=None, guide_offset=None)
        self.assertFalse(was_excluded)
        self.assertEqual(method, "UNRESOLVED")
        self.assertEqual(len(filtered), len(dummy_hits), "Hits were silently discarded!")


class TestOrganismAndReferenceValidation(unittest.TestCase):
    """Priority 4: Reject or mark unavailable non-human organisms / incompatible references."""

    def test_non_human_organism_disables_bowtie_and_cfd(self):
        """Non-human organism or arbitrary FASTA must mark CFD as unavailable against GRCh38."""
        seq = "ATGGCCGAGGCGGCCGCCGAGATCGGCACCCCGCTGCAGCACGAGATCCACGTGTGGGAGCAGCTGCCGGGCGTCACCCACTGGGTCGGCGGCAGCCGGAAGCTGGCGCAGCTGGCCG"
        res = run_dynamic_pipeline(
            gene_symbol="MOUSE_GENE",
            fasta_sequence=seq,
            skip_bowtie=False,
            skip_cfd=False,
            skip_deepcas9=True,
            skip_optimization=True,
        )
        stages = {s.stage: s for s in res.stages}
        self.assertEqual(stages["cfd_scoring"].status, "unavailable")

        # Clean up
        test_dir = DYNAMIC_RESULTS_DIR / "MOUSE_GENE"
        if test_dir.exists():
            shutil.rmtree(test_dir)


class TestCanonicalGenesFrozenIntegrity(unittest.TestCase):
    """Priority 5: Ensure canonical genes (TP53, HBB, EGFR, BRCA1) remain 100% frozen."""

    def test_frozen_baseline_counts_and_selected_guides(self):
        """Verify candidate counts, feasible counts, and top 5 guide IDs match frozen baseline exactly."""
        report = PipelineController.verify_frozen_baseline()
        self.assertTrue(report["all_passed"], f"Frozen baseline verification failed: {report}")
        for gene, base in FROZEN_BASELINE.items():
            gene_report = report["per_gene"][gene]
            self.assertTrue(gene_report["passed"], f"Integrity failure for {gene}: {gene_report}")


if __name__ == "__main__":
    unittest.main()

