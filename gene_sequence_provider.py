"""
gene_sequence_provider.py

Root-level re-export for gene sequence provider abstraction.
"""

from src.gene_sequence_provider import (
    SequenceResolutionError,
    SequenceProvider,
    LocalSequenceProvider,
    NCBISequenceProvider,
    get_default_provider,
    set_default_provider,
    resolve_gene_sequence,
)

__all__ = [
    "SequenceResolutionError",
    "SequenceProvider",
    "LocalSequenceProvider",
    "NCBISequenceProvider",
    "get_default_provider",
    "set_default_provider",
    "resolve_gene_sequence",
]
