"""
src/dynamic_pipeline.py

Dynamic Gene Analysis Pipeline for CRISPR Guide RNA Optimization.
==================================================================

Implements genuine dynamic analysis for genes beyond the canonical four
(TP53, HBB, EGFR, BRCA1) by reusing the validated research pipeline modules.

Pipeline stages:
  1. Sequence Resolution  — NCBI Entrez retrieval via NCBISequenceProvider
  2. Candidate Generation — PAM scanning and guide extraction
  3. Feature Extraction   — GC content, thermodynamics, structure, Bowtie off-targets
  4. DeepSpCas9 Scoring   — On-target activity prediction via loaded model
  5. CFD Off-Target Scoring — Aggregate CFD specificity from Bowtie alignments
  6. Feasibility Filtering — Biological constraint filtering
  7. Multi-Objective Optimization — Pareto analysis and guide selection

Each stage reports its status and availability. If a required resource
(model checkpoint, Bowtie index, genome FASTA) is unavailable, the stage
is marked as "unavailable" with an explanation. No heuristic or proxy
scores are ever substituted.

Dynamic results are isolated from canonical four-gene datasets.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dynamic_pipeline")

# Canonical genes are never processed through the dynamic pipeline
CANONICAL_GENES = {"TP53", "HBB", "EGFR", "BRCA1"}

# Resource paths
DEEPCAS9_WEIGHTS = PROJECT_ROOT / "models" / "deep_spcas9" / "DeepCas9_weights.pt"
BOWTIE_INDEX_PREFIX = PROJECT_ROOT / "data" / "hg38_index" / "hg38_index"
DYNAMIC_RESULTS_DIR = PROJECT_ROOT / "results" / "dynamic"


def _sanitize_value(val: Any) -> Any:
    """Recursively replace NaN and infinite float values with None for RFC 8259 JSON compliance."""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize_value(v) for v in val]
    elif isinstance(val, np.ndarray):
        return _sanitize_value(val.tolist())
    return val


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: str
    status: str  # "completed", "unavailable", "failed", "skipped"
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DynamicPipelineResult:
    """Complete result from the dynamic analysis pipeline."""
    gene_symbol: str
    input_type: str  # "gene_symbol", "accession", "fasta"
    stages: List[StageResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidates_df: Optional[pd.DataFrame] = None
    features_df: Optional[pd.DataFrame] = None
    feasible_df: Optional[pd.DataFrame] = None
    pareto_df: Optional[pd.DataFrame] = None
    selected_guides: Optional[pd.DataFrame] = None

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable API response with RFC 8259 compliant nulls for uncomputed values."""
        stages_summary = []
        for s in self.stages:
            stages_summary.append({
                "stage": s.stage,
                "status": s.status,
                "message": s.message,
                "data": _sanitize_value(s.data),
            })
            if s.error:
                stages_summary[-1]["error"] = s.error

        # Determine overall scoring support status
        deep_spcas9_available = any(
            s.stage == "deepspcas9_scoring" and s.status == "completed"
            for s in self.stages
        )
        cfd_available = any(
            s.stage == "cfd_scoring" and s.status == "completed"
            for s in self.stages
        )
        bowtie_available = any(
            (s.stage == "feature_extraction" and s.data.get("bowtie_available", False)) or
            (s.stage == "cfd_scoring" and s.status == "completed")
            for s in self.stages
        )
        optimization_complete = any(
            s.stage == "optimization" and s.status == "completed"
            for s in self.stages
        )
        is_partial = not (deep_spcas9_available and cfd_available and optimization_complete)

        # Build genuine overview metrics
        overview = {**self.metadata}
        if self.candidates_df is not None:
            overview["total_candidates"] = len(self.candidates_df)
            overview["plus_strand"] = int((self.candidates_df["strand"] == "+").sum())
            overview["minus_strand"] = int((self.candidates_df["strand"] == "-").sum())
        if self.features_df is not None and not self.features_df.empty:
            fdf = self.features_df
            overview["mean_gc"] = round(float(fdf["gc_content"].mean()), 2) if "gc_content" in fdf.columns else None
            overview["mean_tm"] = round(float(fdf["melting_temp_c"].mean()), 2) if "melting_temp_c" in fdf.columns else None
            overview["mean_delta_g"] = round(float(fdf["delta_g_kcal_mol"].mean()), 2) if "delta_g_kcal_mol" in fdf.columns else None
            overview["mean_mfe"] = round(float(fdf["folding_energy_kcal_mol"].mean()), 2) if "folding_energy_kcal_mol" in fdf.columns else None

            if "deep_spcas9_score" in fdf.columns and fdf["deep_spcas9_score"].notna().any():
                act = fdf["deep_spcas9_score"].dropna()
                overview["deep_mean"] = round(float(act.mean()), 2)
                overview["deep_min"] = round(float(act.min()), 2)
                overview["deep_max"] = round(float(act.max()), 2)
                overview["deep_median"] = round(float(act.median()), 2)
            else:
                overview["deep_mean"] = None
                overview["deep_min"] = None
                overview["deep_max"] = None
                overview["deep_median"] = None

            if "cfd_scoring_complete" in fdf.columns and fdf["cfd_scoring_complete"].notna().any():
                overview["complete_searches"] = int((fdf["cfd_scoring_complete"] == True).sum())
                overview["censored_searches"] = int((fdf["cfd_scoring_complete"] == False).sum())
            else:
                overview["complete_searches"] = None
                overview["censored_searches"] = None

            if "cfd_off_target_risk" in fdf.columns and fdf["cfd_off_target_risk"].notna().any():
                r_valid = fdf["cfd_off_target_risk"].dropna()
                overview["max_cfd_risk"] = round(float(r_valid.max()), 2) if not r_valid.empty else None
            else:
                overview["max_cfd_risk"] = None

        if self.feasible_df is not None:
            overview["feasible"] = len(self.feasible_df)
            overview["feasible_candidates"] = len(self.feasible_df)
            if self.candidates_df is not None:
                overview["excluded"] = len(self.candidates_df) - len(self.feasible_df)
        if self.pareto_df is not None:
            overview["pareto"] = len(self.pareto_df)
            overview["pareto_front_size"] = len(self.pareto_df)
        if self.selected_guides is not None:
            overview["selected"] = len(self.selected_guides)
            if "guide_id" in self.selected_guides.columns:
                overview["selected_guide_ids"] = self.selected_guides["guide_id"].tolist()

        # Build response with partial vs complete differentiation
        response = {
            "status": "success",
            "gene_symbol": self.gene_symbol,
            "input_type": self.input_type,
            "dynamic_scoring_supported": True,
            "validated": False,  # Dynamic results are never "validated" like canonical
            "partial_analysis": is_partial,
            "complete_optimization": optimization_complete,
            "scoring_availability": {
                "deepspcas9": deep_spcas9_available,
                "bowtie_off_target": bowtie_available,
                "cfd_specificity": cfd_available,
                "feature_extraction": any(
                    s.stage == "feature_extraction" and s.status == "completed"
                    for s in self.stages
                ),
                "optimization": optimization_complete,
            },
            "stages_available": {s.stage: (s.status == "completed") for s in self.stages},
            "overview": _sanitize_value(overview),
            "meta": _sanitize_value(overview),
            "stages": stages_summary,
        }

        # Include results if available (limit for API response size)
        if self.feasible_df is not None and not self.feasible_df.empty:
            sample = self.feasible_df.head(2000)
            response["feasible_sample"] = _sanitize_value(sample.to_dict(orient="records"))
        else:
            response["feasible_sample"] = []

        if self.pareto_df is not None and not self.pareto_df.empty:
            response["pareto_front"] = _sanitize_value(self.pareto_df.to_dict(orient="records"))
        else:
            response["pareto_front"] = []

        if self.selected_guides is not None and not self.selected_guides.empty:
            response["selected_guides"] = _sanitize_value(self.selected_guides.to_dict(orient="records"))
        elif self.feasible_df is not None and not self.feasible_df.empty:
            response["selected_guides"] = _sanitize_value(self.feasible_df.head(5).to_dict(orient="records"))
        else:
            response["selected_guides"] = []

        # Build human-readable message
        completed_stages = [s.stage for s in self.stages if s.status == "completed"]
        unavailable_stages = [s.stage for s in self.stages if s.status == "unavailable"]
        failed_stages = [s.stage for s in self.stages if s.status == "failed"]

        msg_parts = []
        if completed_stages:
            msg_parts.append(f"Completed stages: {', '.join(completed_stages)}")
        if unavailable_stages:
            msg_parts.append(f"Unavailable: {', '.join(unavailable_stages)}")
        if failed_stages:
            msg_parts.append(f"Failed: {', '.join(failed_stages)}")

        response["message"] = ". ".join(msg_parts) + "."
        response["disclaimer"] = (
            "Dynamic analysis results are generated using the same validated modules "
            "as the canonical research pipeline, but have not undergone manual expert "
            "review. Use these results as computational guidance, not as validated "
            "experimental recommendations."
        )

        return response


def _check_bowtie_available() -> bool:
    """Check if the hg38 Bowtie index is present."""
    index_file = Path(f"{BOWTIE_INDEX_PREFIX}.1.ebwt")
    return index_file.exists()


def _check_deepcas9_available() -> bool:
    """Check if the DeepSpCas9 model weights are present."""
    return DEEPCAS9_WEIGHTS.exists()


_BOWTIE_EXECUTABLE_CACHE: Optional[bool] = None


def _check_bowtie_executable() -> bool:
    """Check if bowtie is callable (via WSL or native)."""
    global _BOWTIE_EXECUTABLE_CACHE
    if _BOWTIE_EXECUTABLE_CACHE is not None:
        return _BOWTIE_EXECUTABLE_CACHE

    import subprocess
    try:
        # Try WSL bowtie (allow 60s for WSL startup if cold)
        result = subprocess.run(
            ["wsl", "bowtie", "--version"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            _BOWTIE_EXECUTABLE_CACHE = True
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["bowtie", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            _BOWTIE_EXECUTABLE_CACHE = True
            return True
    except Exception:
        pass

    _BOWTIE_EXECUTABLE_CACHE = False
    return False


def run_dynamic_pipeline(
    gene_symbol: str,
    accession: Optional[str] = None,
    fasta_path: Optional[str] = None,
    fasta_sequence: Optional[str] = None,
    skip_bowtie: bool = False,
    skip_deepcas9: bool = False,
    skip_cfd: bool = False,
    skip_optimization: bool = False,
    max_mismatches: int = 3,
    threads: int = 4,
) -> DynamicPipelineResult:
    """
    Run the genuine dynamic analysis pipeline for a novel gene.

    Accepts gene symbols, NCBI accessions, or user-provided FASTA sequences.
    Each pipeline stage checks for resource availability and reports its status.
    No heuristic or proxy scores are ever substituted.

    Args:
        gene_symbol: Gene symbol (e.g. "KRAS")
        accession: Optional NCBI accession (e.g. "NM_004985.5")
        fasta_path: Optional path to a FASTA file
        fasta_sequence: Optional raw FASTA/DNA sequence string
        skip_bowtie: Skip Bowtie off-target alignment
        skip_deepcas9: Skip DeepSpCas9 scoring
        skip_cfd: Skip CFD off-target scoring
        skip_optimization: Skip multi-objective optimization
        max_mismatches: Maximum mismatches for Bowtie alignment
        threads: Number of threads for Bowtie

    Returns:
        DynamicPipelineResult with all stage results and data.
    """
    gene_key = gene_symbol.strip().upper()
    result = DynamicPipelineResult(gene_symbol=gene_key, input_type="gene_symbol")

    if gene_key in CANONICAL_GENES and not fasta_path and not fasta_sequence:
        raise ValueError(
            f"Gene '{gene_key}' is a canonical research gene. "
            f"Use PipelineController for validated canonical data."
        )

    # Set up output directory
    output_dir = DYNAMIC_RESULTS_DIR / gene_key
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 72)
    log.info("DYNAMIC PIPELINE: %s", gene_key)
    log.info("=" * 72)

    # ===================================================================
    # Stage 1: Sequence Resolution
    # ===================================================================
    resolved_fasta_path = None
    sequence = None

    if fasta_sequence:
        # User-provided sequence
        result.input_type = "fasta_sequence"
        try:
            # Write to temp FASTA
            temp_fasta = output_dir / f"{gene_key}_user_input.fasta"
            with open(temp_fasta, "w", encoding="utf-8") as fp:
                fp.write(f">{gene_key}_user_provided\n{fasta_sequence}\n")
            resolved_fasta_path = temp_fasta
            sequence = fasta_sequence.upper().replace("\n", "").replace("\r", "")

            result.metadata = {
                "gene_symbol": gene_key,
                "organism": "User-provided",
                "assembly": "N/A",
                "sequence_length": len(sequence),
                "source": "user_input",
            }
            result.stages.append(StageResult(
                stage="sequence_resolution",
                status="completed",
                message=f"User-provided sequence ({len(sequence):,} bp)",
                data={"sequence_length": len(sequence), "source": "user_input"},
            ))
        except Exception as exc:
            result.stages.append(StageResult(
                stage="sequence_resolution",
                status="failed",
                message=f"Failed to process user-provided sequence: {exc}",
                error=traceback.format_exc(),
            ))
            return result

    elif fasta_path:
        # User-provided FASTA file
        result.input_type = "fasta"
        try:
            from src.guide_generation import read_fasta_sequence
            fasta_p = Path(fasta_path)
            if not fasta_p.exists():
                raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
            sequence = read_fasta_sequence(str(fasta_p))
            resolved_fasta_path = fasta_p

            result.metadata = {
                "gene_symbol": gene_key,
                "organism": "User-provided",
                "assembly": "N/A",
                "sequence_length": len(sequence),
                "source": "fasta_file",
                "fasta_path": str(fasta_p),
            }
            result.stages.append(StageResult(
                stage="sequence_resolution",
                status="completed",
                message=f"FASTA file loaded ({len(sequence):,} bp)",
                data={"sequence_length": len(sequence), "source": "fasta_file"},
            ))
        except Exception as exc:
            result.stages.append(StageResult(
                stage="sequence_resolution",
                status="failed",
                message=f"Failed to read FASTA file: {exc}",
                error=traceback.format_exc(),
            ))
            return result

    else:
        # Resolve via NCBI
        if accession:
            result.input_type = "accession"
        try:
            from src.ncbi_sequence_provider import NCBISequenceProvider
            from src.guide_generation import read_fasta_sequence

            provider = NCBISequenceProvider(
                cache_dir=str(PROJECT_ROOT / "data" / "reference_sequences"),
            )

            query = accession if accession else gene_key
            ncbi_result = provider.resolve_with_metadata(query)
            meta = ncbi_result.metadata

            resolved_fasta_path = ncbi_result.fasta_path
            sequence = read_fasta_sequence(str(resolved_fasta_path))

            result.metadata = {
                "gene_symbol": meta.gene_symbol or gene_key,
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
                "source": "ncbi_entrez",
            }
            result.stages.append(StageResult(
                stage="sequence_resolution",
                status="completed",
                message=(
                    f"Resolved {gene_key} from NCBI Entrez: "
                    f"{meta.organism}, {meta.assembly}, {meta.genomic_accession}, "
                    f"{meta.sequence_length:,} bp"
                ),
                data={
                    "ncbi_gene_id": meta.ncbi_gene_id,
                    "accession": meta.genomic_accession,
                    "organism": meta.organism,
                    "assembly": meta.assembly,
                    "sequence_length": meta.sequence_length,
                },
            ))

        except Exception as exc:
            from src.ncbi_sequence_provider import NCBIResolutionError, NCBINetworkError
            if isinstance(exc, (NCBIResolutionError, NCBINetworkError)):
                raise
            result.stages.append(StageResult(
                stage="sequence_resolution",
                status="failed",
                message=f"Failed to resolve gene '{gene_key}' via NCBI: {exc}",
                error=traceback.format_exc(),
            ))
            return result

    # ===================================================================
    # Stage 2: Candidate Generation (PAM scanning)
    # ===================================================================
    try:
        from src.guide_generation import build_guide_dataframe

        candidates_df = build_guide_dataframe(sequence)
        total_candidates = len(candidates_df)

        # Save candidates
        candidates_csv = output_dir / "candidate_gRNAs.csv"
        candidates_df.to_csv(candidates_csv, index=False)

        result.candidates_df = candidates_df
        result.stages.append(StageResult(
            stage="candidate_generation",
            status="completed",
            message=f"Generated {total_candidates:,} candidate guide RNAs",
            data={
                "total_candidates": total_candidates,
                "plus_strand": int((candidates_df["strand"] == "+").sum()),
                "minus_strand": int((candidates_df["strand"] == "-").sum()),
            },
        ))

        if total_candidates == 0:
            result.stages.append(StageResult(
                stage="feature_extraction",
                status="skipped",
                message="No candidates found; skipping downstream stages",
            ))
            return result

    except Exception as exc:
        result.stages.append(StageResult(
            stage="candidate_generation",
            status="failed",
            message=f"PAM scanning failed: {exc}",
            error=traceback.format_exc(),
        ))
        return result

    # ===================================================================
    # Stage 3: Feature Extraction (GC, thermodynamics, structure, Bowtie)
    # ===================================================================
    from src.step9c_cfd import GENOME_FASTA_PATHS
    genome_available = any(os.path.exists(gp) for gp in GENOME_FASTA_PATHS)

    # Organism and reference validation (Priority 4)
    organism = result.metadata.get("organism", "")
    is_human = (organism in ("", "Homo sapiens", "User-provided")) and (result.metadata.get("assembly") in ("GRCh38", "N/A"))
    if not is_human and organism:
        log.warning("Organism '%s' is not human GRCh38; Bowtie/CFD against human genome will be disabled", organism)

    bowtie_index = None
    bowtie_available = False

    if not skip_bowtie and is_human:
        if _check_bowtie_available():
            if _check_bowtie_executable():
                bowtie_index = str(BOWTIE_INDEX_PREFIX)
                bowtie_available = True
            else:
                log.warning("Bowtie index found but bowtie executable not available")
        else:
            log.warning("hg38 Bowtie index not found at %s", BOWTIE_INDEX_PREFIX)

    # Single-pass Bowtie optimization:
    # If CFD is requested and both Bowtie and the reference genome are available for human genomic input,
    # defer Bowtie alignment to Stage 5 so that Bowtie is run once and extracts PAMs in a single pass.
    run_bowtie_in_stage3 = bowtie_available and (skip_cfd or not genome_available or result.input_type in ("fasta_sequence", "fasta"))
    stage3_bowtie_index = bowtie_index if run_bowtie_in_stage3 else None

    try:
        from src.feature_extraction import FeatureConfig, extract_all_features

        feature_config = FeatureConfig(
            input_csv=str(output_dir / "candidate_gRNAs.csv"),
            output_csv=str(output_dir / "gRNA_features.csv"),
            bowtie_index=stage3_bowtie_index,
            max_mismatches=max_mismatches,
            threads=threads,
            defer_bowtie_to_cfd=(not run_bowtie_in_stage3 and bowtie_available),
        )

        features_df = extract_all_features(candidates_df, feature_config, gene_name=gene_key)
        features_df.to_csv(output_dir / "gRNA_features.csv", index=False)

        result.features_df = features_df
        msg = f"Extracted features for {len(features_df):,} candidates."
        if not is_human and organism:
            msg += f" Bowtie off-target unavailable: organism '{organism}' is incompatible with human GRCh38 index."
        elif not bowtie_available:
            msg += " Bowtie off-target: unavailable (specificity columns are uncomputed)."
        elif run_bowtie_in_stage3:
            msg += " Bowtie off-target: available."
        else:
            msg += " Bowtie genome alignment active (deferred to CFD stage for single-pass PAM extraction)."

        result.stages.append(StageResult(
            stage="feature_extraction",
            status="completed",
            message=msg,
            data={
                "n_features": len(features_df.columns),
                "bowtie_available": bowtie_available,
                "single_pass_deferred": not run_bowtie_in_stage3 and bowtie_available,
                "gc_mean": round(float(features_df["gc_content"].mean()), 4) if "gc_content" in features_df.columns else None,
            },
        ))
    except Exception as exc:
        result.stages.append(StageResult(
            stage="feature_extraction",
            status="failed",
            message=f"Feature extraction failed: {exc}",
            error=traceback.format_exc(),
        ))
        return result

    # ===================================================================
    # Stage 4: DeepSpCas9 On-Target Scoring
    # ===================================================================
    if skip_deepcas9:
        result.stages.append(StageResult(
            stage="deepspcas9_scoring",
            status="skipped",
            message="DeepSpCas9 scoring was explicitly skipped",
        ))
    elif not _check_deepcas9_available():
        result.stages.append(StageResult(
            stage="deepspcas9_scoring",
            status="unavailable",
            message=(
                f"DeepSpCas9 model weights not found at {DEEPCAS9_WEIGHTS}. "
                f"Run convert_weights.py to generate the weights file. "
                f"The deep_spcas9_score column will contain uncomputed values."
            ),
        ))
    else:
        try:
            from src.deep_spcas9 import DeepSpCas9Predictor

            predictor = DeepSpCas9Predictor(weights_path=str(DEEPCAS9_WEIGHTS))
            scores, stats = predictor.score_candidates_dataframe(
                result.features_df, sequence
            )
            result.features_df["deep_spcas9_score"] = scores
            result.features_df.to_csv(output_dir / "gRNA_features.csv", index=False)

            result.stages.append(StageResult(
                stage="deepspcas9_scoring",
                status="completed",
                message=(
                    f"DeepSpCas9 predicted {stats['successfully_predicted']}/{stats['total_candidates']} candidates. "
                    f"Score range: [{stats.get('min', 'N/A')}, {stats.get('max', 'N/A')}]"
                ),
                data=stats,
            ))
        except Exception as exc:
            result.stages.append(StageResult(
                stage="deepspcas9_scoring",
                status="failed",
                message=f"DeepSpCas9 scoring failed: {exc}",
                error=traceback.format_exc(),
            ))

    # ===================================================================
    # Stage 5: CFD Off-Target Scoring
    # ===================================================================
    if skip_cfd:
        result.stages.append(StageResult(
            stage="cfd_scoring",
            status="skipped",
            message="CFD scoring was explicitly skipped",
        ))
    elif not is_human and organism:
        result.stages.append(StageResult(
            stage="cfd_scoring",
            status="unavailable",
            message=(
                f"CFD specificity scoring requires an organism-matched reference genome. "
                f"Configured reference index is GRCh38 (Homo sapiens), incompatible with '{organism}'."
            ),
        ))
    elif result.input_type in ("fasta_sequence", "fasta"):
        result.stages.append(StageResult(
            stage="cfd_scoring",
            status="unavailable",
            message=(
                "CFD off-target scoring requires verified genomic locus coordinates on GRCh38 to perform "
                "on-target exclusion. For arbitrary FASTA sequences without genomic coordinates, CFD specificity "
                "is unavailable to avoid misrepresenting specificity."
            ),
        ))
    elif not bowtie_available:
        result.stages.append(StageResult(
            stage="cfd_scoring",
            status="unavailable",
            message=(
                "CFD off-target scoring requires Bowtie alignment data, which is unavailable. "
                "The cfd_off_target_risk and cfd_specificity_score columns will contain uncomputed values."
            ),
        ))
    elif not genome_available:
        result.stages.append(StageResult(
            stage="cfd_scoring",
            status="unavailable",
            message=(
                "Full CFD scoring requires the GRCh38 genome FASTA for PAM extraction. "
                "Genome FASTA not found. CFD specificity scores cannot be computed without the reference genome."
            ),
        ))
    else:
        try:
            _run_full_cfd_scoring(result, output_dir, gene_key, threads)
        except Exception as exc:
            result.stages.append(StageResult(
                stage="cfd_scoring",
                status="failed",
                message=f"CFD scoring failed: {exc}",
                error=traceback.format_exc(),
            ))

    # Ensure CFD columns exist in features_df even when CFD scoring was skipped/unavailable
    if result.features_df is not None:
        if "cfd_specificity_score" not in result.features_df.columns:
            result.features_df["cfd_specificity_score"] = np.nan
        if "cfd_off_target_risk" not in result.features_df.columns:
            result.features_df["cfd_off_target_risk"] = np.nan
        result.features_df.to_csv(output_dir / "gRNA_features.csv", index=False)

    # ===================================================================
    # Stage 6: Feasibility Filtering + Multi-Objective Optimization
    # ===================================================================
    if skip_optimization:
        result.stages.append(StageResult(
            stage="optimization",
            status="skipped",
            message="Optimization was explicitly skipped",
        ))
    else:
        try:
            _run_optimization(result, output_dir, gene_key)
        except Exception as exc:
            result.stages.append(StageResult(
                stage="optimization",
                status="failed",
                message=f"Optimization failed: {exc}",
                error=traceback.format_exc(),
            ))

    # ===================================================================
    # Save pipeline summary
    # ===================================================================
    try:
        api_resp = result.to_api_response()
        summary = {
            "gene_symbol": gene_key,
            "input_type": result.input_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "message": s.message,
                    "data": s.data,
                }
                for s in result.stages
            ],
            "metadata": result.metadata,
            "overview": api_resp.get("overview", {}),
        }
        with open(output_dir / "dynamic_run_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, default=str)
    except Exception:
        pass

    log.info("=" * 72)
    log.info("DYNAMIC PIPELINE COMPLETE: %s", gene_key)
    for s in result.stages:
        log.info("  [%s] %s: %s", s.status.upper(), s.stage, s.message)
    log.info("=" * 72)

    return result


def _run_full_cfd_scoring(
    result: DynamicPipelineResult,
    output_dir: Path,
    gene_key: str,
    threads: int,
) -> None:
    """Run full CFD scoring pipeline using step9c_cfd logic."""
    from pyfaidx import Fasta
    from src.step9c_cfd import (
        parse_bowtie_and_extract_cfd_features,
        score_guide_cfd,
        exclude_on_target,
        GENOME_FASTA_PATHS,
    )

    # Open genome
    genome = None
    for gp in GENOME_FASTA_PATHS:
        if os.path.exists(gp):
            genome = Fasta(gp)
            break

    if genome is None:
        result.stages.append(StageResult(
            stage="cfd_scoring",
            status="unavailable",
            message="GRCh38 genome FASTA not found for CFD PAM extraction",
        ))
        return

    df = result.features_df
    guides = df["guide_rna"].tolist()
    index_prefix = str(BOWTIE_INDEX_PREFIX)

    # Run Bowtie + extract CFD features
    guide_alignments = parse_bowtie_and_extract_cfd_features(
        guides, index_prefix, genome, threads=threads
    )

    # Metadata for intended on-target locus exclusion
    gene_meta = result.metadata if (
        result.metadata.get("genomic_accession") and result.metadata.get("retrieved_start") is not None
    ) else None

    # Compute genuine CFD scores using score_guide_cfd from step9c
    cfd_records = []
    ot_records = []
    need_ot_counts = "off_target_count" not in df.columns or df["off_target_count"].isna().all()
    n_scored = 0

    for _, row in df.iterrows():
        g_seq = row["guide_rna"]
        hits = guide_alignments.get(g_seq, [])
        capped = len(hits) >= 1000 or (
            "off_target_capped" in row and bool(row["off_target_capped"]) and not pd.isna(row["off_target_capped"])
        )

        metrics = score_guide_cfd(
            guide_seq=g_seq,
            hits=hits,
            guide_chrom=str(row.get("chrom", "")) if pd.notna(row.get("chrom")) else None,
            guide_offset=int(row["start"]) if ("start" in row and pd.notna(row["start"])) else None,
            guide_strand=str(row.get("strand", "")) if pd.notna(row.get("strand")) else None,
            gene_meta=gene_meta,
            bowtie_capped=capped,
        )
        metrics["guide_rna"] = g_seq
        if metrics.get("cfd_scoring_complete", False):
            n_scored += 1
        cfd_records.append(metrics)

        if need_ot_counts:
            hits_after, _, _ = exclude_on_target(
                hits,
                guide_chrom=str(row.get("chrom", "")) if pd.notna(row.get("chrom")) else None,
                guide_offset=int(row["start"]) if ("start" in row and pd.notna(row["start"])) else None,
                guide_strand=str(row.get("strand", "")) if pd.notna(row.get("strand")) else None,
                gene_meta=gene_meta,
            )
            ot_0 = sum(1 for h in hits_after if h.get("n_mm_protospacer", 0) == 0)
            ot_1 = sum(1 for h in hits_after if h.get("n_mm_protospacer", 0) == 1)
            ot_2 = sum(1 for h in hits_after if h.get("n_mm_protospacer", 0) == 2)
            ot_3 = sum(1 for h in hits_after if h.get("n_mm_protospacer", 0) == 3)
            ot_total = len(hits_after)
            mm_list = [h.get("n_mm_protospacer", 0) for h in hits_after]
            mm_score = float(np.mean(mm_list)) if mm_list else np.nan

            ot_records.append({
                "guide_rna": g_seq,
                "off_targets_0mm": ot_0,
                "off_targets_1mm": ot_1,
                "off_targets_2mm": ot_2,
                "off_targets_3mm": ot_3,
                "off_target_count": ot_total,
                "off_target_capped": capped,
                "mismatch_score": mm_score,
            })

    cfd_df = pd.DataFrame(cfd_records)
    for col in cfd_df.columns:
        if col != "guide_rna":
            df[col] = cfd_df[col].values

    if need_ot_counts and ot_records:
        ot_df = pd.DataFrame(ot_records)
        for col in ot_df.columns:
            if col != "guide_rna":
                df[col] = ot_df[col].values

    result.features_df = df
    df.to_csv(output_dir / "gRNA_features.csv", index=False)

    result.stages.append(StageResult(
        stage="cfd_scoring",
        status="completed",
        message=f"CFD scored {n_scored}/{len(guides)} guides completely using GRCh38 reference",
        data={"n_scored": n_scored, "n_total": len(guides)},
    ))


def _run_optimization(
    result: DynamicPipelineResult,
    output_dir: Path,
    gene_key: str,
) -> None:
    """Run feasibility filtering and multi-objective optimization."""
    df = result.features_df
    if df is None or df.empty:
        result.stages.append(StageResult(
            stage="optimization",
            status="skipped",
            message="No feature data available for optimization",
        ))
        return

    # Check if we have DeepSpCas9 scores (required for optimization)
    has_deep_spcas9 = (
        "deep_spcas9_score" in df.columns and
        df["deep_spcas9_score"].notna().any()
    )

    if not has_deep_spcas9:
        result.stages.append(StageResult(
            stage="optimization",
            status="unavailable",
            message=(
                "Multi-objective optimization requires DeepSpCas9 on-target activity scores. "
                "These scores are unavailable, so optimization cannot proceed."
            ),
        ))
        return

    # Check if we have CFD specificity scores
    has_cfd = (
        "cfd_specificity_score" in df.columns and
        df["cfd_specificity_score"].notna().any()
    )

    # Apply feasibility filters (reuse grna_optimizer logic)
    from src.grna_optimizer import OptimizerConfig, load_and_filter, find_pareto_front
    from src.optimization import min_max_normalize

    # Create optimizer config
    features_csv = output_dir / "gRNA_features.csv"
    opt_dir = output_dir / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)

    config = OptimizerConfig(
        input_csv=str(features_csv),
        output_dir=str(opt_dir),
        capped_guide_policy="exclude" if has_cfd else "retain",
    )

    feasible_df, excluded_df, filter_report = load_and_filter(config)

    if "cfd_specificity_score" not in feasible_df.columns:
        feasible_df["cfd_specificity_score"] = np.nan
    if "cfd_off_target_risk" not in feasible_df.columns:
        feasible_df["cfd_off_target_risk"] = np.nan

    if feasible_df.empty:
        result.stages.append(StageResult(
            stage="optimization",
            status="completed",
            message=(
                f"All {filter_report.get('total_candidates', 0)} candidates were "
                f"excluded by feasibility filters. No guides are feasible."
            ),
            data=filter_report,
        ))
        result.feasible_df = feasible_df
        return

    # Normalization: Activity is always normalized
    feasible_df["norm_activity"] = min_max_normalize(
        feasible_df["deep_spcas9_score"].values, maximize=True
    )

    if has_cfd:
        # Step 10B CFD-Aware multi-objective optimization
        # Conservative floor for any missing/censored in feasible set
        cfd_spec = feasible_df["cfd_specificity_score"].copy().fillna(0.0)
        feasible_df["norm_cfd_specificity"] = min_max_normalize(
            cfd_spec.values, maximize=True
        )
        spec_axis = feasible_df["norm_cfd_specificity"].values
        spec_metric_name = "cfd_specificity_score"
    else:
        # Fall back to Bowtie off-target count only if available, clearly labeled
        if "off_target_count" in feasible_df.columns and feasible_df["off_target_count"].notna().any():
            ot_counts = feasible_df["off_target_count"].values
            feasible_df["norm_specificity"] = min_max_normalize(
                ot_counts, maximize=False
            )
            spec_axis = feasible_df["norm_specificity"].values
            spec_metric_name = "bowtie_off_target_count (legacy)"
        else:
            if "deep_spcas9_score" in feasible_df.columns and feasible_df["deep_spcas9_score"].notna().any():
                feasible_df = feasible_df.sort_values("deep_spcas9_score", ascending=False)
            result.feasible_df = feasible_df
            result.selected_guides = feasible_df.head(5).copy()
            result.pareto_df = pd.DataFrame()
            feasible_df.to_csv(opt_dir / "feasible_candidates.csv", index=False)
            result.selected_guides.to_csv(opt_dir / "selected_guides.csv", index=False)
            result.stages.append(StageResult(
                stage="optimization",
                status="unavailable",
                message=(
                    f"{len(feasible_df)} feasible candidates found and scored with genuine DeepSpCas9, "
                    f"but multi-objective optimization requires genome-wide specificity data "
                    f"(CFD or Bowtie off-target counts), which is unavailable in this environment. "
                    f"Showing candidates ranked by predicted on-target cleavage activity."
                ),
                data=filter_report,
            ))
            return

    # Discrete Pareto front calculation (exact Track B logic)
    act_axis = feasible_df["norm_activity"].values
    is_pareto = find_pareto_front(act_axis, spec_axis)

    feasible_df["is_pareto"] = is_pareto
    pareto_df = feasible_df[is_pareto].copy()

    # Balance score for guide selection (equal 50/50 weighting)
    balance_scores = 0.5 * act_axis + 0.5 * spec_axis
    feasible_df["balance_score"] = balance_scores
    pareto_df["balance_score"] = feasible_df.loc[is_pareto, "balance_score"].values

    if len(pareto_df) > 0:
        selected = pareto_df.nlargest(min(5, len(pareto_df)), "balance_score").copy()
    else:
        selected = feasible_df.nlargest(min(5, len(feasible_df)), "norm_activity").copy()

    feasible_df["is_selected"] = feasible_df.index.isin(selected.index)

    # Save results
    feasible_df.to_csv(opt_dir / "feasible_candidates.csv", index=False)
    pareto_df.to_csv(opt_dir / "pareto_front.csv", index=False)
    selected.to_csv(opt_dir / "selected_guides.csv", index=False)

    result.feasible_df = feasible_df
    result.pareto_df = pareto_df
    result.selected_guides = selected

    result.stages.append(StageResult(
        stage="optimization",
        status="completed",
        message=(
            f"Feasibility: {len(feasible_df)}/{filter_report.get('total_candidates', 0)} candidates pass constraints. "
            f"Pareto front: {len(pareto_df)} non-dominated guides. "
            f"Selected: {len(selected)} top balanced guides using {spec_metric_name}."
        ),
        data={
            **filter_report,
            "feasible_candidates": len(feasible_df),
            "pareto_front_size": len(pareto_df),
            "selected_count": len(selected),
            "specificity_metric": spec_metric_name,
            "has_cfd_specificity": has_cfd,
        },
    ))
