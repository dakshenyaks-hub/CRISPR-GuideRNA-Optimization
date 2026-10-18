"""
gene_sequence_provider.py

Abstraction layer for resolving gene identifiers to sequence data and FASTA files.
Decouples sequence acquisition (local reference files, NCBI, Ensembl) from
downstream PAM scanning, feature extraction, and optimization pipelines.

Provider hierarchy:
    SequenceProvider (ABC)
    ├── LocalSequenceProvider   — resolves from local file registry
    └── NCBISequenceProvider    — resolves via NCBI Entrez (genomic DNA)

The default provider chain tries NCBI first, then falls back to local files.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from Bio import SeqIO

log = logging.getLogger("gene_sequence_provider")


class SequenceResolutionError(Exception):
    """Raised when a gene sequence cannot be resolved."""
    pass


class SequenceProvider(ABC):
    """Abstract base class for gene sequence providers."""

    @abstractmethod
    def resolve(self, gene_name: str) -> Path:
        """
        Resolve a gene symbol/name to a local FASTA file path.

        Args:
            gene_name: Gene symbol (e.g. 'TP53')

        Returns:
            Path to the resolved FASTA file.
        """
        pass

    @abstractmethod
    def get_sequence(self, gene_name: str) -> str:
        """
        Retrieve the uppercase nucleotide sequence string for the given gene.

        Args:
            gene_name: Gene symbol (e.g. 'TP53')

        Returns:
            DNA sequence string (upper-case)
        """
        pass


class LocalSequenceProvider(SequenceProvider):
    """
    Resolves gene symbols using a registry of local reference FASTA files.
    """

    DEFAULT_GENE_MAP: Dict[str, Path] = {
        "TP53": Path("data/gene.fna"),
    }

    def __init__(
        self,
        gene_map: Optional[Dict[str, Path | str]] = None,
        base_dir: Optional[Path | str] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self.gene_map: Dict[str, Path] = {}

        mapping = gene_map if gene_map is not None else self.DEFAULT_GENE_MAP
        for gene, rel_or_abs_path in mapping.items():
            self.register_gene(gene, rel_or_abs_path)

    def register_gene(self, gene_name: str, file_path: Path | str) -> None:
        """Register or update a local FASTA file path for a gene symbol."""
        path = Path(file_path)
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        self.gene_map[gene_name.strip().upper()] = path

    def resolve(self, gene_name: str) -> Path:
        gene_key = gene_name.strip().upper()
        if gene_key not in self.gene_map:
            available = ", ".join(self.gene_map.keys()) or "None"
            raise SequenceResolutionError(
                f"Gene '{gene_name}' not found in local sequence registry. "
                f"Available local genes: [{available}]"
            )
        fasta_path = self.gene_map[gene_key]
        if not fasta_path.exists():
            raise FileNotFoundError(
                f"Local FASTA file for gene '{gene_name}' does not exist at path: {fasta_path}"
            )
        return fasta_path

    def get_sequence(self, gene_name: str) -> str:
        fasta_path = self.resolve(gene_name)
        record = next(SeqIO.parse(str(fasta_path), "fasta"), None)
        if record is None:
            raise ValueError(f"No FASTA records found in: {fasta_path}")
        return str(record.seq).upper()


class NCBISequenceProvider(SequenceProvider):
    """
    Resolves gene symbols via NCBI Entrez to cached local FASTA files
    containing genomic DNA from the GRCh38 reference assembly.

    This is a thin adapter that delegates to the implementation in
    src/ncbi_sequence_provider.py.
    """

    def __init__(self, **kwargs) -> None:
        # Lazy import to avoid circular imports and heavy init
        from src.ncbi_sequence_provider import (
            NCBISequenceProvider as _NCBIImpl,
        )
        self._impl = _NCBIImpl(**kwargs)

    def resolve(self, gene_name: str) -> Path:
        return self._impl.resolve(gene_name)

    def get_sequence(self, gene_name: str) -> str:
        return self._impl.get_sequence(gene_name)

    def resolve_with_metadata(self, gene_name: str):
        """Return full SequenceResult with metadata."""
        return self._impl.resolve_with_metadata(gene_name)


# ---------------------------------------------------------------------------
# Default provider management
# ---------------------------------------------------------------------------

# Default singleton — now uses NCBI with local fallback
_default_provider: SequenceProvider = NCBISequenceProvider()


def get_default_provider() -> SequenceProvider:
    """Return the currently configured default sequence provider."""
    return _default_provider


def set_default_provider(provider: SequenceProvider) -> None:
    """Set the global default sequence provider."""
    global _default_provider
    _default_provider = provider


def resolve_gene_sequence(
    gene_name: str,
    provider: Optional[SequenceProvider] = None,
) -> Path:
    """
    Resolve a gene symbol to its local FASTA path using the active provider.

    Args:
        gene_name: Target gene name (e.g. 'TP53')
        provider: Optional custom SequenceProvider. Defaults to global provider.

    Returns:
        Path to local FASTA file.
    """
    active_provider = provider or _default_provider
    return active_provider.resolve(gene_name)
