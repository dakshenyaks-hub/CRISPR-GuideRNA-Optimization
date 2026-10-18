"""
tests/test_ncbi_sequence_provider.py

Unit tests for NCBI sequence retrieval provider, gene symbol validation,
and caching mechanisms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ncbi_sequence_provider import (
    NCBISequenceProvider,
    SequenceMetadata,
    SequenceResult,
    NCBIResolutionError,
    NCBINetworkError,
    NCBISequenceUnsuitableError,
)
from src.serve_dashboard import _is_valid_gene_symbol, _resolve_gene_ncbi


# ---------------------------------------------------------------------------
# Test Gene Symbol Validation
# ---------------------------------------------------------------------------

class TestGeneSymbolValidation:
    """Test _is_valid_gene_symbol helper."""

    def test_valid_standard_symbols(self):
        for sym in ["TP53", "BRCA1", "EGFR", "HBB", "VEGFA", "MYC", "KRAS", "IL6"]:
            assert _is_valid_gene_symbol(sym) is True

    def test_valid_with_hyphen_and_dots(self):
        assert _is_valid_gene_symbol("HLA-A") is True
        assert _is_valid_gene_symbol("MIR-21") is True
        assert _is_valid_gene_symbol("NKX2.1") is True

    def test_invalid_symbols(self):
        assert _is_valid_gene_symbol("") is False
        assert _is_valid_gene_symbol("   ") is False
        assert _is_valid_gene_symbol("TP53; DROP TABLE") is False
        assert _is_valid_gene_symbol("<script>") is False
        assert _is_valid_gene_symbol("A" * 35) is False
        assert _is_valid_gene_symbol("-INVALID") is False


# ---------------------------------------------------------------------------
# Test SequenceMetadata & SequenceResult
# ---------------------------------------------------------------------------

class TestDataStructures:
    """Test metadata serialization and dataclasses."""

    def test_metadata_to_dict(self):
        meta = SequenceMetadata(
            gene_symbol="TP53",
            ncbi_gene_id="7157",
            chromosome="17",
            genomic_accession="NC_000017.11",
            assembly="GRCh38",
            sequence_length=15000,
        )
        d = meta.to_dict()
        assert d["gene_symbol"] == "TP53"
        assert d["ncbi_gene_id"] == "7157"
        assert d["genomic_accession"] == "NC_000017.11"
        assert d["sequence_length"] == 15000

    def test_sequence_result(self, tmp_path):
        fasta_file = tmp_path / "test.fa"
        fasta_file.write_text(">test\nACGTACGT\n")
        meta = SequenceMetadata(gene_symbol="TEST", sequence_length=8)
        res = SequenceResult(
            fasta_path=fasta_file,
            gene_name="TEST",
            accession="NC_000001.11",
            organism="Homo sapiens",
            assembly="GRCh38",
            sequence_length=8,
            metadata=meta,
        )
        assert res.fasta_path == fasta_file
        assert res.gene_name == "TEST"


# ---------------------------------------------------------------------------
# Test NCBISequenceProvider Local Cache
# ---------------------------------------------------------------------------

class TestNCBISequenceProviderCache:
    """Test local cache hit and miss handling."""

    def test_init_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "seq_cache"
        provider = NCBISequenceProvider(cache_dir=str(cache_dir))
        assert Path(provider.cache_dir).exists()

    def test_cache_hit_returns_existing(self, tmp_path):
        cache_dir = tmp_path / "seq_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        fasta_path = cache_dir / "TEST_GENE_NC_000001.11.fasta"
        meta_path = cache_dir / "TEST_GENE_NC_000001.11.metadata.json"
        
        fasta_path.write_text(">TEST_GENE Homo sapiens GRCh38\nACGTACGTACGT\n")
        meta_dict = {
            "gene_symbol": "TEST_GENE",
            "organism": "Homo sapiens",
            "ncbi_gene_id": "99999",
            "chromosome": "1",
            "genomic_accession": "NC_000001.11",
            "assembly": "GRCh38",
            "sequence_type": "genomic_DNA",
            "strand": "plus",
            "gene_start": 100,
            "gene_stop": 200,
            "retrieved_start": 50,
            "retrieved_stop": 250,
            "sequence_length": 12,
            "retrieval_timestamp": "2026-10-07T00:00:00Z",
            "fasta_path": str(fasta_path),
            "selection_rule": "cached",
            "notes": "unit test",
        }
        meta_path.write_text(json.dumps(meta_dict))

        provider = NCBISequenceProvider(cache_dir=str(cache_dir))
        result = provider.resolve_with_metadata("TEST_GENE")

        assert result.gene_name == "TEST_GENE"
        assert result.metadata.ncbi_gene_id == "99999"
        assert result.metadata.sequence_length == 12


# ---------------------------------------------------------------------------
# Test _resolve_gene_ncbi with mocked provider
# ---------------------------------------------------------------------------

class TestResolveGeneNCBI:
    """Test _resolve_gene_ncbi wrapper function used by dashboard."""

    def test_resolve_with_valid_sequence(self, tmp_path):
        cache_dir = tmp_path / "seq_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = cache_dir / "MOCKG_genomic.fasta"
        
        mock_seq = "A" * 20 + "AGG" + "C" * 20 + "TGG" + "T" * 20
        fasta_path.write_text(f">MOCKG\n{mock_seq}\n")

        mock_meta = SequenceMetadata(
            gene_symbol="MOCKG",
            ncbi_gene_id="12345",
            chromosome="2",
            genomic_accession="NC_000002.12",
            assembly="GRCh38",
            strand="plus",
            gene_start=1000,
            gene_stop=2000,
            retrieved_start=900,
            retrieved_stop=2100,
            sequence_length=len(mock_seq),
            retrieval_timestamp="2026-10-07T00:00:00Z",
            fasta_path=str(fasta_path),
            notes="mock test",
        )
        mock_res = SequenceResult(
            fasta_path=fasta_path,
            gene_name="MOCKG",
            accession="NC_000002.12",
            organism="Homo sapiens",
            assembly="GRCh38",
            sequence_length=len(mock_seq),
            metadata=mock_meta,
        )

        with patch("src.ncbi_sequence_provider.NCBISequenceProvider.resolve_with_metadata", return_value=mock_res):
            out = _resolve_gene_ncbi("MOCKG")
            assert out["status"] == "success"
            assert out["gene_symbol"] == "MOCKG"
            assert out["ncbi_gene_id"] == "12345"
            assert out["total_candidates"] >= 0
            assert "plus_strand" in out
            assert "minus_strand" in out
