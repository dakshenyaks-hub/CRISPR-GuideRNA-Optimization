"""
ncbi_sequence_provider.py

NCBI-backed sequence provider for retrieving human genomic DNA sequences
suitable for CRISPR guide RNA design.

Strategy:
    1. Use NCBI Entrez esearch on the 'gene' database to find the Gene ID
       for a given human gene symbol.
    2. Use efetch with rettype='gene_table' and esummary to retrieve
       the gene's chromosomal coordinates and RefSeq chromosome accession
       (e.g. NC_000017.11 for chromosome 17 on GRCh38).
    3. Use efetch on 'nuccore' with seq_start/seq_stop to download the
       exact genomic region in FASTA format.
    4. Cache the result locally with full provenance metadata.

This module deliberately retrieves GENOMIC sequence (not CDS or mRNA),
because CRISPR guide RNAs target genomic DNA including introns, UTRs,
and flanking regions.

Dependencies:
    - Biopython (Bio.Entrez, Bio.SeqIO)  — already in requirements.txt
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import socket
from xml.etree import ElementTree

from Bio import Entrez, SeqIO

log = logging.getLogger("ncbi_sequence_provider")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NCBI requires an email for Entrez usage
_ENTREZ_EMAIL = os.environ.get("NCBI_EMAIL", "crispr-grna-tool@example.com")
_ENTREZ_API_KEY = os.environ.get("NCBI_API_KEY", None)
_ENTREZ_TOOL = "CRISPR-GuideRNA-Optimization"

# Default cache directory (relative to project root)
_DEFAULT_CACHE_DIR = Path("data/reference_sequences")

# Flanking bases to add upstream/downstream of the gene body.
# This ensures we capture promoter-proximal PAM sites and full UTRs.
_FLANKING_BP = 2000

# Default network timeout for NCBI queries in seconds
_DEFAULT_TIMEOUT = 15.0

# We restrict to Homo sapiens for now.
_ORGANISM = "Homo sapiens"
_TAXID = "9606"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SequenceMetadata:
    """Full provenance record for a retrieved genomic sequence."""

    gene_symbol: str = ""
    organism: str = _ORGANISM
    ncbi_gene_id: str = ""
    chromosome: str = ""
    genomic_accession: str = ""  # e.g. NC_000017.11
    assembly: str = "GRCh38"
    sequence_type: str = "genomic_DNA"
    strand: str = ""                # plus / minus
    gene_start: int = 0            # 1-based, on the chromosome
    gene_stop: int = 0
    retrieved_start: int = 0       # actual fetched window (with flanking)
    retrieved_stop: int = 0
    sequence_length: int = 0
    retrieval_timestamp: str = ""
    fasta_path: str = ""
    selection_rule: str = ""       # how/why this record was chosen
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SequenceResult:
    """Structured result returned by NCBISequenceProvider.resolve()."""

    fasta_path: Path
    gene_name: str
    accession: str
    organism: str
    assembly: str
    sequence_length: int
    metadata: SequenceMetadata


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NCBIResolutionError(Exception):
    """Raised when a gene cannot be resolved via NCBI."""
    pass


class NCBINetworkError(Exception):
    """Raised when NCBI cannot be reached."""
    pass


class NCBISequenceUnsuitableError(Exception):
    """Raised when the retrieved sequence is not suitable for CRISPR design."""
    pass


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class NCBISequenceProvider:
    """
    Resolves gene symbols to cached local FASTA files containing genomic DNA
    from the GRCh38 reference assembly via NCBI Entrez.

    Implements the SequenceProvider interface contract:
        resolve(gene_name) -> Path   (to a FASTA file)
        get_sequence(gene_name) -> str (uppercase DNA string)

    Additionally exposes:
        resolve_with_metadata(gene_name) -> SequenceResult
    """

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout) if timeout is not None else _DEFAULT_TIMEOUT
        socket.setdefaulttimeout(self.timeout)

        Entrez.email = email or _ENTREZ_EMAIL
        Entrez.tool = _ENTREZ_TOOL
        if api_key or _ENTREZ_API_KEY:
            Entrez.api_key = api_key or _ENTREZ_API_KEY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, gene_name: str) -> Path:
        """Resolve a gene symbol to a local FASTA path (downloading if needed)."""
        result = self.resolve_with_metadata(gene_name)
        return result.fasta_path

    def get_sequence(self, gene_name: str) -> str:
        """Return the uppercase DNA sequence string for the given gene."""
        fasta_path = self.resolve(gene_name)
        record = next(SeqIO.parse(str(fasta_path), "fasta"), None)
        if record is None:
            raise ValueError(f"No FASTA records found in: {fasta_path}")
        return str(record.seq).upper()

    def resolve_with_metadata(self, gene_name: str) -> SequenceResult:
        """
        Full resolution: returns SequenceResult with path + metadata.
        Uses cache if available.
        """
        gene_key = gene_name.strip().upper()

        # Check cache first
        cached = self._find_cached(gene_key)
        if cached is not None:
            log.info(
                "Using cached sequence for %s: %s", gene_key, cached.fasta_path
            )
            return cached

        # Fetch from NCBI
        log.info("No cached sequence found for %s — querying NCBI...", gene_key)
        return self._fetch_from_ncbi(gene_key)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _find_cached(self, gene_key: str) -> Optional[SequenceResult]:
        """Look for an existing cached FASTA + metadata for this gene."""
        meta_pattern = f"{gene_key}_*.metadata.json"
        meta_files = sorted(self.cache_dir.glob(meta_pattern))

        for meta_file in meta_files:
            try:
                with open(meta_file, "r") as f:
                    meta_dict = json.load(f)
                meta = SequenceMetadata(**meta_dict)
                fasta_path = Path(meta.fasta_path)
                if not fasta_path.exists():
                    local_candidate = self.cache_dir / Path(meta.fasta_path).name
                    if local_candidate.exists():
                        fasta_path = local_candidate
                    else:
                        stem_candidate = self.cache_dir / f"{meta_file.stem}.fasta"
                        if stem_candidate.exists():
                            fasta_path = stem_candidate

                if fasta_path.exists():
                    return SequenceResult(
                        fasta_path=fasta_path,
                        gene_name=meta.gene_symbol,
                        accession=meta.genomic_accession,
                        organism=meta.organism,
                        assembly=meta.assembly,
                        sequence_length=meta.sequence_length,
                        metadata=meta,
                    )
                else:
                    log.warning(
                        "Cache metadata exists but FASTA missing: %s", fasta_path
                    )
            except Exception as exc:
                log.warning("Corrupt cache metadata %s: %s", meta_file, exc)
        return None

    def _save_cache(self, meta: SequenceMetadata) -> None:
        """Persist metadata alongside the cached FASTA."""
        fasta_stem = Path(meta.fasta_path).stem  # e.g. TP53_NC_000017.11
        meta_path = self.cache_dir / f"{fasta_stem}.metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)
        log.info("Saved cache metadata: %s", meta_path)

    # ------------------------------------------------------------------
    # NCBI retrieval
    # ------------------------------------------------------------------

    def _fetch_from_ncbi(self, gene_key: str) -> SequenceResult:
        """
        Multi-step NCBI Entrez workflow:
          1. esearch gene database → Gene ID
          2. esummary gene → genomic location (chr, start, stop, accession)
          3. efetch nuccore → FASTA of the genomic region
        """
        # Step 1: Search for the gene in NCBI Gene database
        gene_id = self._search_gene(gene_key)

        # Step 2: Get gene summary (coordinates, accession)
        gene_info = self._get_gene_summary(gene_id, gene_key)

        # Step 3: Fetch genomic FASTA
        result = self._fetch_genomic_fasta(gene_key, gene_info)

        return result

    def _entrez_call(self, func, **kwargs):
        """Wrapper for Entrez calls with error handling and network timeout."""
        try:
            handle = func(**kwargs)
            return handle
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(
                term in error_msg
                for term in ["urlopen", "urlerror", "timeout", "timed out", "connection", "http"]
            ):
                raise NCBINetworkError(
                    "Could not connect to NCBI or request timed out. Please check your internet "
                    "connection and try again."
                ) from exc
            raise

    def _search_gene(self, gene_symbol: str) -> str:
        """Search NCBI Gene database for a human gene symbol. Returns Gene ID."""
        query = f'{gene_symbol}[Gene Name] AND {_ORGANISM}[Organism] AND alive[property]'
        log.info("NCBI Entrez esearch: db=gene, query=%s", query)

        handle = self._entrez_call(
            Entrez.esearch, db="gene", term=query, retmax=10
        )
        result = Entrez.read(handle)
        handle.close()

        id_list = result.get("IdList", [])
        if not id_list:
            raise NCBIResolutionError(
                f"Gene '{gene_symbol}' not found in NCBI Gene database for "
                f"{_ORGANISM}. Please check the gene symbol and try again.\n"
                f"Search query used: {query}"
            )

        if len(id_list) == 1:
            gene_id = id_list[0]
            log.info("Found unique Gene ID: %s", gene_id)
            return gene_id

        # Multiple hits — apply selection rule
        log.info(
            "Multiple Gene IDs found for '%s': %s — applying selection rules...",
            gene_symbol,
            id_list,
        )
        return self._select_best_gene_id(gene_symbol, id_list)

    def _select_best_gene_id(self, gene_symbol: str, id_list: list[str]) -> str:
        """
        When multiple Gene IDs are returned, select the best one.

        Selection rule (explicit):
          1. Prefer the record whose official symbol exactly matches the query.
          2. Among exact matches, prefer protein-coding genes.
          3. If still tied, prefer the lowest Gene ID (oldest / most established).
        """
        time.sleep(0.34)  # Rate limit
        handle = self._entrez_call(
            Entrez.esummary, db="gene", id=",".join(id_list)
        )
        summaries = Entrez.read(handle)
        handle.close()

        # Entrez.read returns a ListElement of dicts for esummary on gene
        doc_sums = summaries.get("DocumentSummarySet", {}).get(
            "DocumentSummary", []
        )

        candidates = []
        for doc in doc_sums:
            uid = str(doc.attributes.get("uid", ""))
            official_symbol = str(doc.get("NomenclatureSymbol", "")).upper()
            name = str(doc.get("Name", "")).upper()
            description = str(doc.get("Description", ""))
            organism_name = str(doc.get("Organism", {}).get("ScientificName", ""))

            # Check if this is for the correct organism
            if _ORGANISM.lower() not in organism_name.lower():
                continue

            exact_match = official_symbol == gene_symbol.upper() or name == gene_symbol.upper()

            candidates.append({
                "gene_id": uid,
                "symbol": official_symbol or name,
                "description": description,
                "exact_match": exact_match,
            })

        if not candidates:
            raise NCBIResolutionError(
                f"Multiple Gene IDs found for '{gene_symbol}' but none match "
                f"{_ORGANISM}: {id_list}"
            )

        # Sort: exact matches first, then by Gene ID (ascending)
        candidates.sort(key=lambda c: (not c["exact_match"], int(c["gene_id"])))

        selected = candidates[0]
        log.info(
            "Selected Gene ID %s (symbol=%s, description=%s) from %d candidates. "
            "Selection rule: exact symbol match preferred, then lowest Gene ID.",
            selected["gene_id"],
            selected["symbol"],
            selected["description"],
            len(candidates),
        )

        if len(candidates) > 1:
            log.info(
                "Other candidates considered: %s",
                [(c["gene_id"], c["symbol"], c["description"]) for c in candidates[1:]],
            )

        return selected["gene_id"]

    def _get_gene_summary(self, gene_id: str, gene_symbol: str) -> dict:
        """
        Fetch gene annotation from NCBI Gene to determine:
          - Chromosome accession (e.g. NC_000017.11)
          - Genomic start/stop coordinates
          - Strand orientation
          - Chromosome number

        Uses efetch with rettype=docsum in XML format to get the
        GenomicInfo block containing RefSeq chromosome annotations.
        """
        time.sleep(0.34)  # Rate limit

        log.info("NCBI Entrez efetch: db=gene, id=%s, rettype=xml", gene_id)
        handle = self._entrez_call(
            Entrez.efetch, db="gene", id=gene_id, rettype="xml"
        )
        xml_data = handle.read()
        handle.close()

        # Parse the XML to extract genomic location
        root = ElementTree.fromstring(xml_data)

        # Navigate to: Entrezgene > Entrezgene_locus > Gene-commentary
        # We need the genomic RefSeq accession and coordinates
        gene_info = {
            "gene_id": gene_id,
            "symbol": gene_symbol,
            "chromosome": "",
            "accession": "",
            "start": 0,
            "stop": 0,
            "strand": "",
            "assembly": "GRCh38",
        }

        # Extract official symbol
        for gene_ref in root.iter("Gene-ref"):
            locus = gene_ref.find("Gene-ref_locus")
            if locus is not None and locus.text:
                gene_info["symbol"] = locus.text

        # Find the genomic placement on the reference assembly
        # Look for Gene-commentary with type 'genomic' and heading 'Reference GRCh38.p14'
        # The structure is: Gene-commentary (type=genomic, accession=NC_...)
        #   -> Gene-commentary_seqs -> Seq-loc -> Seq-loc_int -> Seq-interval
        found_placement = False

        for commentary in root.iter("Gene-commentary"):
            # Check heading for reference assembly
            heading_elem = commentary.find("Gene-commentary_heading")
            heading = heading_elem.text if heading_elem is not None else ""

            # We want the 'Reference GRCh38' annotation
            if "GRCh38" not in heading:
                continue

            # Look for sub-commentaries with genomic accession
            for sub_commentary in commentary.iter("Gene-commentary"):
                accession_elem = sub_commentary.find("Gene-commentary_accession")
                if accession_elem is None or not accession_elem.text:
                    continue

                acc = accession_elem.text
                # We want the chromosome-level RefSeq accession (NC_*)
                if not acc.startswith("NC_"):
                    continue

                # Get version
                version_elem = sub_commentary.find("Gene-commentary_version")
                if version_elem is not None and version_elem.text:
                    acc = f"{acc}.{version_elem.text}"

                gene_info["accession"] = acc

                # Extract coordinates from Seq-interval
                for seq_int in sub_commentary.iter("Seq-interval"):
                    from_elem = seq_int.find("Seq-interval_from")
                    to_elem = seq_int.find("Seq-interval_to")
                    strand_elem = seq_int.find(
                        ".//Na-strand"
                    )

                    if from_elem is not None and to_elem is not None:
                        # NCBI uses 0-based coordinates in XML
                        gene_info["start"] = int(from_elem.text) + 1  # Convert to 1-based
                        gene_info["stop"] = int(to_elem.text) + 1

                        if strand_elem is not None:
                            strand_val = strand_elem.attrib.get("value", "")
                            gene_info["strand"] = strand_val  # 'plus' or 'minus'

                        found_placement = True
                        break
                if found_placement:
                    break
            if found_placement:
                break

        # Extract chromosome from the accession
        # NC_000017 -> chromosome 17
        if gene_info["accession"]:
            try:
                chr_num = int(gene_info["accession"].split("_")[1].split(".")[0])
                if chr_num <= 22:
                    gene_info["chromosome"] = str(chr_num)
                elif chr_num == 23:
                    gene_info["chromosome"] = "X"
                elif chr_num == 24:
                    gene_info["chromosome"] = "Y"
            except (ValueError, IndexError):
                gene_info["chromosome"] = "unknown"

        if not found_placement or not gene_info["accession"]:
            raise NCBIResolutionError(
                f"Could not determine genomic coordinates for gene '{gene_symbol}' "
                f"(Gene ID: {gene_id}) on the GRCh38 reference assembly.\n"
                f"This may indicate the gene is not placed on a reference chromosome."
            )

        log.info(
            "Gene %s (ID: %s): chr%s:%d-%d (%s strand), accession=%s, assembly=%s",
            gene_info["symbol"],
            gene_id,
            gene_info["chromosome"],
            gene_info["start"],
            gene_info["stop"],
            gene_info["strand"],
            gene_info["accession"],
            gene_info["assembly"],
        )

        return gene_info

    def _fetch_genomic_fasta(self, gene_key: str, gene_info: dict) -> SequenceResult:
        """
        Fetch the genomic DNA sequence from NCBI nuccore using the
        chromosome accession and coordinates from the gene summary.

        Adds flanking regions (_FLANKING_BP) to capture promoter-proximal
        and downstream PAM sites.
        """
        accession = gene_info["accession"]
        gene_start = gene_info["start"]
        gene_stop = gene_info["stop"]
        strand_str = gene_info["strand"]

        # Add flanking sequence
        fetch_start = max(1, gene_start - _FLANKING_BP)
        fetch_stop = gene_stop + _FLANKING_BP

        # For NCBI efetch, seq_start and seq_stop are 1-based
        # strand=1 for plus, strand=2 for minus
        ncbi_strand = 2 if strand_str == "minus" else 1

        log.info(
            "Fetching genomic sequence: accession=%s, region=%d-%d "
            "(gene body %d-%d + %d bp flanking), strand=%s",
            accession,
            fetch_start,
            fetch_stop,
            gene_start,
            gene_stop,
            _FLANKING_BP,
            strand_str or "plus",
        )

        time.sleep(0.34)  # Rate limit

        handle = self._entrez_call(
            Entrez.efetch,
            db="nuccore",
            id=accession,
            rettype="fasta",
            retmode="text",
            seq_start=fetch_start,
            seq_stop=fetch_stop,
            strand=ncbi_strand,
        )
        fasta_text = handle.read()
        handle.close()

        if not fasta_text or not fasta_text.strip():
            raise NCBIResolutionError(
                f"NCBI returned an empty sequence for {accession}:{fetch_start}-{fetch_stop}. "
                f"The accession may have been retired or the coordinates may be invalid."
            )

        # Parse to validate
        from io import StringIO
        records = list(SeqIO.parse(StringIO(fasta_text), "fasta"))

        if not records:
            raise NCBIResolutionError(
                f"Failed to parse FASTA response from NCBI for "
                f"{accession}:{fetch_start}-{fetch_stop}"
            )

        record = records[0]
        seq_len = len(record.seq)

        # Biological validation
        self._validate_for_crispr(record, gene_key, gene_info)

        # Build descriptive FASTA header
        # Match the format of the existing gene.fna for compatibility:
        # >NC_000017.11:c7687490-7668421 TP53 [organism=Homo sapiens] [GeneID=7157] [chromosome=17]
        if strand_str == "minus":
            coord_str = f"c{fetch_stop}-{fetch_start}"
        else:
            coord_str = f"{fetch_start}-{fetch_stop}"

        header = (
            f"{accession}:{coord_str} {gene_info['symbol']} "
            f"[organism={_ORGANISM}] [GeneID={gene_info['gene_id']}] "
            f"[chromosome={gene_info['chromosome']}]"
        )

        # Rewrite FASTA with our descriptive header
        fasta_lines = [f">{header}"]
        seq_str = str(record.seq).upper()
        # Wrap at 70 characters per line (standard FASTA)
        for i in range(0, len(seq_str), 70):
            fasta_lines.append(seq_str[i : i + 70])
        fasta_content = "\n".join(fasta_lines) + "\n"

        # Save to cache
        # Sanitize accession for filename (replace dots)
        safe_accession = accession.replace(".", "_")
        fasta_filename = f"{gene_key}_{safe_accession}.fasta"
        fasta_path = self.cache_dir / fasta_filename
        fasta_path.write_text(fasta_content, encoding="utf-8")
        log.info("Saved genomic FASTA: %s (%d bp)", fasta_path, seq_len)

        # Build metadata
        meta = SequenceMetadata(
            gene_symbol=gene_info["symbol"],
            organism=_ORGANISM,
            ncbi_gene_id=gene_info["gene_id"],
            chromosome=gene_info["chromosome"],
            genomic_accession=accession,
            assembly=gene_info["assembly"],
            sequence_type="genomic_DNA",
            strand=strand_str or "plus",
            gene_start=gene_start,
            gene_stop=gene_stop,
            retrieved_start=fetch_start,
            retrieved_stop=fetch_stop,
            sequence_length=seq_len,
            retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
            fasta_path=str(fasta_path),
            selection_rule=(
                "Retrieved genomic DNA from GRCh38 reference chromosome. "
                "Region includes gene body plus 2000 bp flanking on each side. "
                "Sequence is on the gene's native strand orientation."
            ),
        )

        self._save_cache(meta)

        return SequenceResult(
            fasta_path=fasta_path,
            gene_name=gene_info["symbol"],
            accession=accession,
            organism=_ORGANISM,
            assembly=gene_info["assembly"],
            sequence_length=seq_len,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Biological validation
    # ------------------------------------------------------------------

    def _validate_for_crispr(self, record, gene_symbol: str, gene_info: dict) -> None:
        """
        Validate that the retrieved sequence is suitable for genomic
        CRISPR guide design.

        Checks:
          - Sequence is DNA (not protein)
          - Sequence is long enough to contain PAM sites
          - Sequence contains standard nucleotides
          - Sequence is from a reference chromosome (NC_ accession)
        """
        seq_str = str(record.seq).upper()
        seq_len = len(seq_str)

        # Check 1: Minimum length
        if seq_len < 23:
            raise NCBISequenceUnsuitableError(
                f"Retrieved sequence for '{gene_symbol}' is only {seq_len} bp. "
                f"A minimum of 23 bp (20-nt guide + 3-nt PAM) is required for "
                f"CRISPR guide design. This sequence is too short."
            )

        # Check 2: DNA content (should be mostly ACGT)
        valid_bases = set("ACGTNRYSWKMBDHV")  # IUPAC DNA
        non_dna = set(seq_str) - valid_bases
        if non_dna:
            # Check if it looks like protein
            protein_chars = set("EFILPQ")  # chars in protein but not IUPAC DNA
            if protein_chars & set(seq_str):
                raise NCBISequenceUnsuitableError(
                    f"Retrieved sequence for '{gene_symbol}' appears to be a "
                    f"PROTEIN sequence, not genomic DNA. CRISPR guides must be "
                    f"designed against genomic DNA. Characters found: {non_dna}"
                )
            raise NCBISequenceUnsuitableError(
                f"Retrieved sequence for '{gene_symbol}' contains non-DNA "
                f"characters: {non_dna}. This may not be a valid genomic sequence."
            )

        # Check 3: Not too many Ns (ambiguous bases)
        n_count = seq_str.count("N")
        n_fraction = n_count / seq_len if seq_len > 0 else 0
        if n_fraction > 0.5:
            raise NCBISequenceUnsuitableError(
                f"Retrieved sequence for '{gene_symbol}' has {n_fraction:.1%} "
                f"ambiguous bases (N). This suggests a poorly assembled region "
                f"that is unsuitable for reliable CRISPR guide design."
            )

        # Check 4: Accession should be a RefSeq chromosome (NC_)
        accession = gene_info.get("accession", "")
        if not accession.startswith("NC_"):
            log.warning(
                "Accession '%s' is not a RefSeq chromosome (NC_*). "
                "The sequence may not be from the primary reference assembly.",
                accession,
            )

        log.info(
            "Sequence validation passed: %d bp genomic DNA, "
            "%.1f%% standard bases (ACGT), accession=%s",
            seq_len,
            (1 - n_fraction) * 100,
            accession,
        )
