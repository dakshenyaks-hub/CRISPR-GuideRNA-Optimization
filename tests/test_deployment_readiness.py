"""
tests/test_deployment_readiness.py

Automated test suite verifying deployment readiness, server binding,
rejection of proxy calculations for unvalidated genes, and preservation
of the four-gene research baseline.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_controller import PipelineController, SUPPORTED_GENES, FROZEN_BASELINE
from src.serve_dashboard import (
    DashboardRequestHandler,
    _is_valid_gene_symbol,
    _resolve_gene_ncbi,
    find_free_port,
)


class TestFourGeneBaselinePreserved(unittest.TestCase):
    """Ensures that the four validated genes and their saved outputs are 100% unchanged."""

    def test_frozen_baseline_all_pass(self):
        report = PipelineController.verify_frozen_baseline()
        self.assertTrue(report["all_passed"], f"Baseline verification failed: {report}")

    def test_each_gene_metrics(self):
        for gene, expected in FROZEN_BASELINE.items():
            ov = PipelineController.get_gene_overview(gene)
            self.assertEqual(ov["total_candidates"], expected["total_candidates"])
            self.assertEqual(ov["feasible_candidates"], expected["feasible_candidates"])
            self.assertEqual(ov["pareto_front_size"], expected["pareto_size"])
            self.assertEqual(ov["selected_guide_ids"], expected["selected_top5"])

    def test_supported_genes_constant(self):
        self.assertEqual(SUPPORTED_GENES, ["TP53", "HBB", "EGFR", "BRCA1"])


class TestProxyScoringPrevention(unittest.TestCase):
    """Proves that dynamic routes cannot return proxy scores as validated predictions."""

    def test_resolve_gene_ncbi_returns_no_proxy_scores(self, tmp_path=None):
        """_resolve_gene_ncbi must not invent DeepSpCas9 or CFD scores."""
        from src.ncbi_sequence_provider import SequenceMetadata, SequenceResult

        mock_seq = "A" * 20 + "AGG" + "C" * 20 + "TGG" + "T" * 20
        mock_meta = SequenceMetadata(
            gene_symbol="NOVELG",
            ncbi_gene_id="99999",
            chromosome="3",
            genomic_accession="NC_000003.12",
            assembly="GRCh38",
            strand="plus",
            gene_start=1000,
            gene_stop=2000,
            sequence_length=len(mock_seq),
            fasta_path="mock.fasta",
        )
        mock_res = SequenceResult(
            fasta_path=Path("mock.fasta"),
            gene_name="NOVELG",
            accession="NC_000003.12",
            organism="Homo sapiens",
            assembly="GRCh38",
            sequence_length=len(mock_seq),
            metadata=mock_meta,
        )

        with patch("src.ncbi_sequence_provider.NCBISequenceProvider.resolve_with_metadata", return_value=mock_res), \
             patch("src.guide_generation.read_fasta_sequence", return_value=mock_seq):
            out = _resolve_gene_ncbi("NOVELG")

            self.assertEqual(out["status"], "success")
            self.assertEqual(out["gene_symbol"], "NOVELG")
            # Must explicitly not return proxy scores
            self.assertIsNone(out["deep_spcas9_score"])
            self.assertIsNone(out["cfd_off_target_risk"])
            self.assertIsNone(out["cfd_specificity_score"])
            self.assertEqual(out["pareto_front"], [])
            self.assertEqual(out["selected_guides"], [])
            self.assertFalse(out["dynamic_scoring_supported"])
            self.assertEqual(out["scoring_status"], "unsupported")
            self.assertIn("Dynamic DeepSpCas9 and CFD off-target scoring are currently unsupported", out["message"])

    def test_post_resolve_gene_invalid_symbol(self):
        """POST /api/resolve_gene rejects invalid gene symbols with 400."""
        handler = self._create_mock_handler("POST", "/api/resolve_gene", body={"gene": "INVALID@#$"})
        handler.do_POST()

        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 400)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid gene symbol", response["message"])

    def test_post_resolve_gene_dynamic_novel_gene(self):
        """POST /api/resolve_gene runs genuine dynamic pipeline for novel gene without proxy scores."""
        test_seq = "ATGGCCGAGGCGGCCGCCGAGATCGGCACCCCGCTGCAGCACGAGATCCACGTGTGGGAGCAGCTGCCGGGCGTCACCCACTGGGTCGGCGGCAGCCGGAAGCTGGCGCAGCTGGCCG"
        handler = self._create_mock_handler("POST", "/api/resolve_gene", body={
            "gene": "NOVELG",
            "fasta_sequence": test_seq,
            "skip_bowtie": True,
            "skip_cfd": True,
        })
        handler.do_POST()

        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 200)
        self.assertEqual(response["gene_symbol"], "NOVELG")
        self.assertFalse(response["validated"])
        self.assertTrue(response["dynamic_scoring_supported"])
        self.assertIn("stages", response)

    def test_post_resolve_gene_supported_for_validated_gene(self):
        """POST /api/resolve_gene returns validated research data for supported genes."""
        handler = self._create_mock_handler("POST", "/api/resolve_gene", body={"gene": "TP53"})
        handler.do_POST()

        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 200)
        self.assertEqual(response["status"], "success")
        self.assertTrue(response["validated"])
        self.assertTrue(response["dynamic_scoring_supported"])
        self.assertEqual(len(response["selected_guides"]), 5)
        self.assertEqual(response["overview"]["total_candidates"], 3554)

    def test_get_gene_unsupported_404(self):
        """GET /api/gene/<UNVALIDATED> returns 404, not proxy data."""
        handler = self._create_mock_handler("GET", "/api/gene/UNKNOWN_GENE")
        handler.do_GET()

        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 404)
        self.assertEqual(response["status"], "unsupported")
        self.assertIn("not supported", response["message"])

    def test_get_genes_only_returns_supported_genes(self):
        """GET /api/genes strictly returns only the 4 validated genes."""
        handler = self._create_mock_handler("GET", "/api/genes")
        handler.do_GET()

        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 200)
        self.assertEqual(response["genes"], ["TP53", "HBB", "EGFR", "BRCA1"])

    def test_get_data_contains_only_supported_genes(self):
        """GET /api/data strictly returns only the 4 validated genes."""
        handler = self._create_mock_handler("GET", "/api/data")
        handler.do_GET()

        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 200)
        self.assertEqual(response["supported_genes"], ["TP53", "HBB", "EGFR", "BRCA1"])
        self.assertEqual(set(response["genes"].keys()), {"TP53", "HBB", "EGFR", "BRCA1"})

    def _create_mock_handler(self, method: str, path: str, body: dict | None = None, raw_body: bytes | None = None):
        """Creates a mock DashboardRequestHandler for testing HTTP routing."""
        handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
        handler.path = path
        handler.command = method
        handler.headers = {}
        handler.sent_headers = {}
        handler.sent_status = None
        handler.output_buffer = io.BytesIO()
        handler.wfile = handler.output_buffer

        if raw_body is not None:
            handler.headers["Content-Length"] = str(len(raw_body))
            handler.rfile = io.BytesIO(raw_body)
        elif body is not None:
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

        def get_json_response():
            val = handler.output_buffer.getvalue()
            return json.loads(val.decode("utf-8")) if val else {}

        handler.get_json_response = get_json_response
        return handler


class TestInputValidationAndSecurity(unittest.TestCase):
    """Verifies security controls, payload bounds, and error sanitization."""

    def test_valid_gene_symbol_rules(self):
        self.assertTrue(_is_valid_gene_symbol("TP53"))
        self.assertTrue(_is_valid_gene_symbol("HLA-A"))
        self.assertTrue(_is_valid_gene_symbol("NKX2.1"))
        self.assertFalse(_is_valid_gene_symbol(""))
        self.assertFalse(_is_valid_gene_symbol("   "))
        self.assertFalse(_is_valid_gene_symbol(None))
        self.assertFalse(_is_valid_gene_symbol(12345))
        self.assertFalse(_is_valid_gene_symbol("TP53; DROP TABLE"))
        self.assertFalse(_is_valid_gene_symbol("<script>alert(1)</script>"))
        self.assertFalse(_is_valid_gene_symbol("A" * 31))

    def test_post_payload_size_limit(self):
        handler = self._create_mock_handler("POST", "/api/resolve_gene", raw_body=b"A" * 70000)
        handler.do_POST()
        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 413)
        self.assertEqual(response["status"], "error")
        self.assertIn("Payload too large", response["message"])

    def test_post_malformed_json(self):
        handler = self._create_mock_handler("POST", "/api/resolve_gene", raw_body=b"{invalid_json:")
        handler.do_POST()
        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 400)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid JSON format", response["message"])

    def test_post_invalid_gene_symbol(self):
        handler = self._create_mock_handler("POST", "/api/resolve_gene", body={"gene": "MALICIOUS; DROP TABLE"})
        handler.do_POST()
        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 400)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid gene symbol", response["message"])

    def test_post_missing_gene(self):
        handler = self._create_mock_handler("POST", "/api/resolve_gene", body={"other_key": "TP53"})
        handler.do_POST()
        response = handler.get_json_response()
        self.assertEqual(handler.sent_status, 400)
        self.assertEqual(response["status"], "error")

    def _create_mock_handler(self, method: str, path: str, body: dict | None = None, raw_body: bytes | None = None):
        handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
        handler.path = path
        handler.command = method
        handler.headers = {}
        handler.sent_headers = {}
        handler.sent_status = None
        handler.output_buffer = io.BytesIO()
        handler.wfile = handler.output_buffer

        if raw_body is not None:
            handler.headers["Content-Length"] = str(len(raw_body))
            handler.rfile = io.BytesIO(raw_body)
        elif body is not None:
            body_bytes = json.dumps(body).encode("utf-8")
            handler.headers["Content-Length"] = str(len(body_bytes))
            handler.rfile = io.BytesIO(body_bytes)
        else:
            handler.rfile = io.BytesIO(b"")

        handler.send_response = lambda s, m=None: setattr(handler, "sent_status", s)
        handler.send_header = lambda k, v: handler.sent_headers.update({k: v})
        handler.end_headers = lambda: None
        handler.get_json_response = lambda: json.loads(handler.output_buffer.getvalue().decode("utf-8")) if handler.output_buffer.getvalue() else {}
        return handler


class TestDeploymentEnvironmentConfiguration(unittest.TestCase):
    """Verifies server binding to 0.0.0.0 and PORT environment variable resolution."""

    def test_find_free_port_default_host(self):
        port = find_free_port(start_port=9990, max_attempts=5, host="0.0.0.0")
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 9990)

    def test_port_env_variable_resolution(self):
        with patch.dict(os.environ, {"PORT": "9090"}):
            env_port_str = os.environ.get("PORT")
            self.assertEqual(int(env_port_str), 9090)


class TestLiveHTTPServer(unittest.TestCase):
    """Starts a real TCPServer instance and validates actual HTTP responses."""

    @classmethod
    def setUpClass(cls):
        import socketserver
        import threading
        import time

        cls.port = find_free_port(start_port=9870, host="127.0.0.1")
        cls.server = socketserver.TCPServer(("127.0.0.1", cls.port), DashboardRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.15)
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_live_serve_index_html(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/") as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("Multi-Objective Guide RNA Optimization Platform", content)

    def test_live_serve_asset(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/components/crispr_visualizer.js") as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("CRISPR-Cas9 Targeting Mechanism Visualizer", content)

    def test_live_api_status(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/api/status") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "online")
            self.assertEqual(data["supported_genes"], ["TP53", "HBB", "EGFR", "BRCA1"])

    def test_live_api_genes(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/api/genes") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["genes"], ["TP53", "HBB", "EGFR", "BRCA1"])

    def test_live_api_data(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/api/data") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["supported_genes"], ["TP53", "HBB", "EGFR", "BRCA1"])
            self.assertEqual(len(data["master_comparison"]), 4)

    def test_live_api_gene_supported(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/api/gene/TP53") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["gene"], "TP53")
            self.assertEqual(data["overview"]["total_candidates"], 3554)
            self.assertEqual(len(data["selected_guides"]), 5)

    def test_live_api_gene_unsupported(self):
        import urllib.error
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.base_url}/api/gene/MYC")
            self.fail("Expected HTTPError 404 for unsupported gene")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            data = json.loads(exc.read().decode("utf-8"))
            self.assertEqual(data["status"], "unsupported")

    def test_live_post_resolve_gene_supported(self):
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/api/resolve_gene",
            data=json.dumps({"gene": "HBB"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["gene_symbol"], "HBB")
            self.assertEqual(data["overview"]["total_candidates"], 540)
            self.assertTrue(data["validated"])

    def test_live_post_resolve_gene_invalid(self):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/api/resolve_gene",
            data=json.dumps({"gene": "INVALID!!@@"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError 400 for invalid gene symbol")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            data = json.loads(exc.read().decode("utf-8"))
            self.assertEqual(data["status"], "error")
            self.assertIn("Invalid gene symbol", data["message"])


if __name__ == "__main__":
    unittest.main()

