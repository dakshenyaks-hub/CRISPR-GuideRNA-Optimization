"""
src/serve_dashboard.py

High-Performance Local Server for the Multi-Gene CRISPR Guide RNA Optimization Dashboard.
==========================================================================================

Features:
  - Serves static dashboard assets (HTML, CSS, JS, Plotly).
  - Exposes REST API endpoints powered by src/pipeline_controller.py.
  - Automatically regenerates dashboard/data.json if missing.
  - Supports automatic port selection (default: 8000) and launches web browser.
  - Provides /api/resolve_gene for dynamic NCBI gene retrieval.

Usage:
  python src/serve_dashboard.py
  python src/serve_dashboard.py --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import math
import os
import re
import socket
import socketserver
import sys
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_controller import PipelineController, SUPPORTED_GENES

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DATA_JSON_PATH = DASHBOARD_DIR / "data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serve_dashboard")


def _is_valid_gene_symbol(symbol: str) -> bool:
    """Validate gene symbol: 1–30 chars, alphanumeric + hyphen + dot + underscore."""
    if not isinstance(symbol, str):
        return False
    trimmed = symbol.strip()
    if not trimmed or len(trimmed) > 30:
        return False
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,29}$", trimmed))


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN and infinite float values with None for standard JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _resolve_gene_ncbi(gene_symbol: str) -> dict:
    """
    Resolve a gene symbol via NCBI sequence provider and perform PAM scanning.

    IMPORTANT: Dynamic DeepSpCas9 activity scoring and CFD off-target calculations
    are explicitly NOT executed for novel genes. Proxy or heuristic scores are never
    substituted. Truly validated DeepSpCas9 and CFD predictions are precomputed
    and maintained exclusively for SUPPORTED_GENES.
    """
    from src.ncbi_sequence_provider import NCBISequenceProvider
    from src.guide_generation import build_guide_dataframe, read_fasta_sequence

    provider = NCBISequenceProvider(
        cache_dir=str(PROJECT_ROOT / "data" / "reference_sequences"),
    )

    result = provider.resolve_with_metadata(gene_symbol)
    meta = result.metadata

    meta_dict = {
        "gene_symbol": meta.gene_symbol,
        "ncbi_gene_id": meta.ncbi_gene_id,
        "organism": meta.organism,
        "assembly": meta.assembly,
        "chromosome": meta.chromosome,
        "genomic_accession": meta.genomic_accession,
        "strand": meta.strand,
        "gene_start": meta.gene_start,
        "gene_stop": meta.gene_stop,
        "retrieved_start": meta.retrieved_start,
        "retrieved_stop": meta.retrieved_stop,
        "sequence_length": meta.sequence_length,
        "description": f"{meta.gene_symbol} (Homo sapiens)",
    }

    sequence = read_fasta_sequence(str(result.fasta_path))
    df = build_guide_dataframe(sequence)
    total_candidates = len(df)
    n_plus = int((df["strand"] == "+").sum()) if total_candidates > 0 else 0
    n_minus = int((df["strand"] == "-").sum()) if total_candidates > 0 else 0

    overview = {
        **meta_dict,
        "total_candidates": total_candidates,
        "plus_strand": n_plus,
        "minus_strand": n_minus,
    }

    return {
        "status": "success",
        **meta_dict,
        "overview": overview,
        "meta": meta_dict,
        "total_candidates": total_candidates,
        "plus_strand": n_plus,
        "minus_strand": n_minus,
        "deep_spcas9_score": None,
        "cfd_off_target_risk": None,
        "cfd_specificity_score": None,
        "feasible_sample": [],
        "pareto_front": [],
        "selected_guides": [],
        "dynamic_scoring_supported": False,
        "scoring_status": "unsupported",
        "message": (
            f"Successfully resolved genomic DNA and {total_candidates:,} PAM candidate guides "
            f"for {gene_symbol}. Dynamic DeepSpCas9 and CFD off-target scoring are currently unsupported. "
            f"Validated scoring is available exclusively for: {', '.join(SUPPORTED_GENES)}."
        ),
    }


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler with REST API routing and CORS support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/resolve_gene":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 65536:
                    self.send_json({"status": "error", "message": "Payload too large (maximum 64 KB)."}, status=413)
                    return

                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.send_json({"status": "error", "message": "Invalid JSON format in request body."}, status=400)
                    return

                if not isinstance(payload, dict):
                    self.send_json({"status": "error", "message": "Request body must be a JSON object."}, status=400)
                    return

                raw_gene = payload.get("gene")
                if not raw_gene or not isinstance(raw_gene, str):
                    self.send_json({"status": "error", "message": "Missing or invalid 'gene' field in request body."}, status=400)
                    return

                gene_symbol = raw_gene.strip().upper()
                if not _is_valid_gene_symbol(gene_symbol):
                    self.send_json({"status": "error", "message": f"Invalid gene symbol: '{gene_symbol}'."}, status=400)
                    return

                # If requested gene is one of our four precomputed research genes, return its verified baseline
                if gene_symbol in SUPPORTED_GENES:
                    log.info("Serving verified research data for precomputed gene: %s", gene_symbol)
                    overview = PipelineController.get_gene_overview(gene_symbol)
                    pareto = PipelineController.get_pareto_front(gene_symbol)
                    selected = PipelineController.get_selected_guides(gene_symbol)
                    feasible = PipelineController.get_feasible_candidates(gene_symbol, 2000)
                    meta = PipelineController.get_gene_metadata(gene_symbol)
                    self.send_json({
                        "status": "success",
                        "gene_symbol": gene_symbol,
                        "overview": overview,
                        "meta": meta,
                        "pareto_front": pareto,
                        "selected_guides": selected,
                        "feasible_sample": feasible,
                        "dynamic_scoring_supported": True,
                        "validated": True,
                        "note": f"Validated research dataset loaded for {gene_symbol}."
                    }, status=200)
                    return

                # Dynamic analysis for novel genes using genuine pipeline modules
                log.info("Dynamic gene analysis requested for: %s", gene_symbol)

                # Extract optional parameters
                fasta_sequence = payload.get("fasta_sequence")
                accession = payload.get("accession")
                skip_bowtie = payload.get("skip_bowtie", False)
                skip_deepcas9 = payload.get("skip_deepcas9", False)
                skip_cfd = payload.get("skip_cfd", False)
                skip_optimization = payload.get("skip_optimization", False)

                try:
                    from src.dynamic_pipeline import run_dynamic_pipeline
                    from src.ncbi_sequence_provider import NCBIResolutionError, NCBINetworkError
                    pipeline_result = run_dynamic_pipeline(
                        gene_symbol=gene_symbol,
                        accession=accession,
                        fasta_sequence=fasta_sequence,
                        skip_bowtie=skip_bowtie,
                        skip_deepcas9=skip_deepcas9,
                        skip_cfd=skip_cfd,
                        skip_optimization=skip_optimization,
                    )
                    response = pipeline_result.to_api_response()
                    self.send_json(response, status=200)
                except NCBIResolutionError as exc:
                    log.warning("Gene not found via NCBI: %s", exc)
                    self.send_json({
                        "status": "error",
                        "gene_symbol": gene_symbol,
                        "dynamic_scoring_supported": False,
                        "message": str(exc),
                        "supported_genes": list(SUPPORTED_GENES),
                    }, status=404)
                except NCBINetworkError as exc:
                    log.warning("NCBI network error: %s", exc)
                    self.send_json({
                        "status": "error",
                        "gene_symbol": gene_symbol,
                        "dynamic_scoring_supported": False,
                        "message": str(exc),
                        "supported_genes": list(SUPPORTED_GENES),
                    }, status=503)
                except Exception as exc:
                    log.error("Dynamic pipeline failed for %s: %s", gene_symbol, exc, exc_info=True)
                    self.send_json({
                        "status": "error",
                        "gene_symbol": gene_symbol,
                        "dynamic_scoring_supported": False,
                        "message": (
                            f"Dynamic analysis failed for gene '{gene_symbol}': {exc}. "
                            f"Validated data is available for: {', '.join(SUPPORTED_GENES)}."
                        ),
                        "supported_genes": list(SUPPORTED_GENES),
                    }, status=500)
                return

            except Exception as exc:
                log.error("Internal error in POST /api/resolve_gene: %s", exc, exc_info=True)
                self.send_json({"status": "error", "message": "An internal error occurred while processing the request."}, status=500)
            return

        self.send_error(404, f"POST endpoint not found: {path}")

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/status":
                from src.dynamic_pipeline import _check_deepcas9_available, _check_bowtie_available, _check_bowtie_executable
                self.send_json({
                    "status": "online",
                    "platform": "CRISPR-Optimization-Dashboard",
                    "supported_genes": list(SUPPORTED_GENES),
                    "dynamic_analysis_supported": True,
                    "resources": {
                        "deepspcas9_model": _check_deepcas9_available(),
                        "hg38_bowtie_index": _check_bowtie_available(),
                        "bowtie_executable": _check_bowtie_executable(),
                    },
                })
                return

            if path == "/api/genes":
                self.send_json({
                    "genes": list(SUPPORTED_GENES),
                    "dynamic_genes": PipelineController.list_dynamic_genes(),
                })
                return

            if path == "/api/dynamic_genes":
                self.send_json({
                    "dynamic_genes": PipelineController.list_dynamic_genes(),
                })
                return

            if path == "/api/data":
                payload = PipelineController.get_full_dashboard_payload()
                self.send_json(payload)
                return

            if path.startswith("/api/gene/"):
                gene = path.split("/")[-1].upper()
                if gene in SUPPORTED_GENES:
                    data = {
                        "gene": gene,
                        "validated": True,
                        "overview": PipelineController.get_gene_overview(gene),
                        "pareto_front": PipelineController.get_pareto_front(gene),
                        "selected_guides": PipelineController.get_selected_guides(gene),
                        "feasible_sample": PipelineController.get_feasible_candidates(gene, 2000),
                        "sensitivity": PipelineController.get_sensitivity_summary(gene),
                    }
                    self.send_json(data)
                    return
                else:
                    # Check if it was analyzed dynamically
                    dyn_summary = PipelineController.get_dynamic_gene_summary(gene)
                    if dyn_summary is not None:
                        # Load dynamic results
                        dyn_dir = PROJECT_ROOT / "results" / "dynamic" / gene
                        opt_dir = dyn_dir / "optimization"
                        pareto = []
                        selected = []
                        feasible = []
                        if (opt_dir / "pareto_front.csv").exists():
                            pareto = pd.read_csv(opt_dir / "pareto_front.csv").to_dict(orient="records")
                        if (opt_dir / "selected_guides.csv").exists():
                            selected = pd.read_csv(opt_dir / "selected_guides.csv").to_dict(orient="records")
                        if (opt_dir / "feasible_candidates.csv").exists():
                            feasible = pd.read_csv(opt_dir / "feasible_candidates.csv").head(2000).to_dict(orient="records")
                        self.send_json({
                            "gene": gene,
                            "validated": False,
                            "dynamic": True,
                            "summary": dyn_summary,
                            "pareto_front": pareto,
                            "selected_guides": selected,
                            "feasible_sample": feasible,
                        })
                        return
                    self.send_json({
                        "status": "unsupported",
                        "gene": gene,
                        "message": f"Gene '{gene}' is not supported or loaded. Validated data is available for: {', '.join(SUPPORTED_GENES)}. Use POST /api/resolve_gene to run dynamic analysis."
                    }, status=404)
                    return

            # /api/ncbi_gene_info/<GENE> — quick metadata-only check (no NCBI call if not cached)
            if path.startswith("/api/ncbi_gene_info/"):
                gene_symbol = path.split("/")[-1].upper()
                if not _is_valid_gene_symbol(gene_symbol):
                    self.send_json({"status": "error", "message": "Invalid gene symbol."}, status=400)
                    return
                cache_dir = PROJECT_ROOT / "data" / "reference_sequences"
                meta_files = sorted(cache_dir.glob(f"{gene_symbol}_*.metadata.json")) if cache_dir.exists() else []
                if meta_files:
                    try:
                        with open(meta_files[-1], "r", encoding="utf-8") as fp:
                            cached_meta = json.load(fp)
                        self.send_json({"status": "cached", "gene_symbol": gene_symbol, "metadata": cached_meta})
                        return
                    except Exception:
                        pass
                self.send_json({"status": "not_cached", "gene_symbol": gene_symbol})
                return

            # Fallback to static file serving
            return super().do_GET()
        except Exception as exc:
            log.error("Internal error in GET %s: %s", self.path, exc, exc_info=True)
            self.send_json({"status": "error", "message": "An internal error occurred while processing the request."}, status=500)

    def send_json(self, data: dict, status: int = 200):
        try:
            clean_data = _sanitize_for_json(data)
            body = json.dumps(clean_data, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass


def find_free_port(start_port: int = 8000, max_attempts: int = 20, host: str = "0.0.0.0") -> int:
    """Finds an available local TCP port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start_port


def ensure_data_json():
    """Generates dashboard/data.json if missing."""
    if not DATA_JSON_PATH.exists():
        log.info("dashboard/data.json not found. Generating from canonical outputs...")
        payload = PipelineController.get_full_dashboard_payload()
        with open(DATA_JSON_PATH, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        log.info("Saved dashboard/data.json successfully.")


def main():
    parser = argparse.ArgumentParser(description="Serve CRISPR Guide RNA Optimization Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to serve dashboard on (default: $PORT or 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--regenerate-data", action="store_true", help="Force regenerate dashboard/data.json")
    args = parser.parse_args()

    # Priority: CLI --port > environment variable PORT > fallback free port
    env_port_str = os.environ.get("PORT")
    env_port = int(env_port_str) if env_port_str and env_port_str.isdigit() else None

    if args.port is not None:
        port = args.port
    elif env_port is not None:
        port = env_port
    else:
        port = find_free_port(8000, host=args.host)

    is_cloud_env = bool(env_port_str)

    if args.regenerate_data or not DATA_JSON_PATH.exists():
        log.info("Regenerating dashboard/data.json from pipeline controller...")
        payload = PipelineController.get_full_dashboard_payload()
        with open(DATA_JSON_PATH, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        log.info("Data regenerated successfully.")

    host = args.host
    url = f"http://{'localhost' if host in ('0.0.0.0', '') else host}:{port}"

    log.info("=" * 72)
    log.info("CRISPR MULTI-GENE OPTIMIZATION DASHBOARD SERVER")
    log.info("=" * 72)
    log.info("  Host binding:      %s", host)
    log.info("  Port:              %d", port)
    log.info("  Serving directory: %s", DASHBOARD_DIR)
    log.info("  Dashboard URL:     %s", url)
    log.info("  Supported genes:   %s", ", ".join(SUPPORTED_GENES))
    log.info("=" * 72)

    if not args.no_browser and not is_cloud_env:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server_address = (host, port)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(server_address, DashboardRequestHandler) as httpd:
        try:
            log.info("Server running at http://%s:%d. Press Ctrl+C to stop.", host, port)
            httpd.serve_forever()
        except KeyboardInterrupt:
            log.info("\nServer stopped.")


if __name__ == "__main__":
    main()

