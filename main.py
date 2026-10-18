"""
main.py

Entry point and orchestrator for the CRISPR Guide RNA Optimization Framework.
Resolves target genes to sequence data, generates candidate gRNAs, and
computes the biological feature table.

Usage:
    python main.py --gene TP53
    python main.py --gene TP53 --bowtie-index data/hg38_index/hg38_index
    python main.py --gene TP53 --skip-features
    python main.py --fasta data/gene.fna
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gene_sequence_provider import (
    resolve_gene_sequence,
    SequenceResolutionError,
    get_default_provider,
    NCBISequenceProvider,
)
from src.guide_generation import run_pipeline, read_fasta_sequence
from src.feature_extraction import FeatureConfig
from src.feature_extraction import run_pipeline as run_feature_pipeline
from src.deep_spcas9 import DeepSpCas9Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRISPR Guide RNA Optimization Framework - Candidate Generation & Feature Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gene",
        type=str,
        help="Target gene symbol to optimize guides for (e.g., TP53)",
    )
    group.add_argument(
        "--fasta",
        type=str,
        help="Path to direct target FASTA file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination CSV path for candidate guide RNAs (defaults to results/<GENE>/candidate_gRNAs.csv)",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        default=None,
        help="Destination CSV path for the feature table (defaults to results/<GENE>/gRNA_features.csv)",
    )
    parser.add_argument(
        "--bowtie-index",
        type=str,
        default=None,
        help="Prebuilt Bowtie index prefix for off-target specificity (e.g., data/hg38_index/hg38_index)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        default=False,
        help="Skip feature extraction (only generate guide candidates)",
    )
    parser.add_argument(
        "--skip-specificity",
        action="store_true",
        default=False,
        help="Skip Bowtie off-target specificity analysis (specificity columns will be NaN)",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=3,
        help="Maximum mismatches for Bowtie off-target alignment",
    )
    parser.add_argument(
        "--skip-deepcas9",
        action="store_true",
        default=False,
        help="Skip DeepSpCas9 on-target efficiency prediction (deep_spcas9_score will be NaN)",
    )
    parser.add_argument(
        "--deepcas9-weights",
        type=str,
        default=None,
        help="Custom path to converted DeepSpCas9 weights file (.pt)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for Bowtie alignment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gene_name = ""
    seq_meta = None

    # Determine default output directory based on target
    if args.gene:
        gene_name = args.gene.strip().upper()
        default_out_dir = Path(f"results/{gene_name}")
    else:
        fasta_stem = Path(args.fasta).stem
        default_out_dir = Path(f"results/{fasta_stem}")

    candidate_output_path = Path(args.output) if args.output else default_out_dir / "candidate_gRNAs.csv"
    features_output_path = Path(args.features_output) if args.features_output else default_out_dir / "gRNA_features.csv"
    summary_json_path = default_out_dir / "run_summary.json"

    # ===================================================================
    # Step 1: Sequence Resolution
    # ===================================================================
    if args.gene:
        log.info("Resolving gene sequence for target: %s", gene_name)
        try:
            provider = get_default_provider()

            # If the provider supports metadata, use it for richer logging
            if isinstance(provider, NCBISequenceProvider):
                result = provider.resolve_with_metadata(args.gene)
                fasta_path = result.fasta_path
                seq_meta = result.metadata

                log.info("=" * 72)
                log.info("SEQUENCE RESOLUTION SUMMARY")
                log.info("-" * 72)
                log.info("  Gene symbol:       %s", seq_meta.gene_symbol)
                log.info("  Organism:          %s", seq_meta.organism)
                log.info("  NCBI Gene ID:      %s", seq_meta.ncbi_gene_id)
                log.info("  Chromosome:        %s", seq_meta.chromosome)
                log.info("  Genomic accession: %s", seq_meta.genomic_accession)
                log.info("  Reference assembly:%s", seq_meta.assembly)
                log.info("  Sequence type:     %s", seq_meta.sequence_type)
                log.info("  Strand:            %s", seq_meta.strand)
                log.info("  Gene coordinates:  %d-%d", seq_meta.gene_start, seq_meta.gene_stop)
                log.info("  Retrieved region:  %d-%d", seq_meta.retrieved_start, seq_meta.retrieved_stop)
                log.info("  Sequence length:   %d bp", seq_meta.sequence_length)
                log.info("  FASTA path:        %s", seq_meta.fasta_path)
                if seq_meta.retrieval_timestamp:
                    log.info("  Retrieved at:      %s", seq_meta.retrieval_timestamp)
                log.info("=" * 72)
            else:
                fasta_path = resolve_gene_sequence(args.gene)
                log.info("Resolved gene '%s' to FASTA path: %s", args.gene, fasta_path)

        except SequenceResolutionError as err:
            log.error("Failed to resolve gene '%s': %s", args.gene, err)
            sys.exit(1)
        except FileNotFoundError as err:
            log.error("Failed to resolve gene '%s': %s", args.gene, err)
            sys.exit(1)
        except Exception as err:
            error_type = type(err).__name__
            log.error(
                "Failed to resolve gene '%s' (%s): %s", args.gene, error_type, err
            )
            sys.exit(1)
    else:
        fasta_path = Path(args.fasta).resolve()
        if not fasta_path.exists():
            log.error("Provided FASTA file does not exist: %s", fasta_path)
            sys.exit(1)
        log.info("Using explicit FASTA path: %s", fasta_path)

    # ===================================================================
    # Step 2: Ensure output directory exists
    # ===================================================================
    candidate_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Step 3: Run PAM/guide generation pipeline
    # ===================================================================
    log.info("Executing PAM scanning and gRNA candidate generation pipeline...")
    candidates_df = run_pipeline(str(fasta_path), output_csv=str(candidate_output_path))

    log.info(
        "Candidate generation complete: %d guide candidates generated and saved to %s",
        len(candidates_df),
        candidate_output_path,
    )

    # ===================================================================
    # Step 4: Feature Extraction
    # ===================================================================
    if args.skip_features:
        log.info("Skipping feature extraction (--skip-features flag set).")
        return

    log.info("")
    log.info("=" * 72)
    log.info("FEATURE EXTRACTION")
    log.info("=" * 72)

    # Configure Bowtie specificity
    bowtie_index = None
    if not args.skip_specificity:
        if args.bowtie_index:
            bowtie_index = args.bowtie_index
        else:
            # Auto-detect the default index location
            default_index = Path("data/hg38_index/hg38_index")
            if (default_index.parent / f"{default_index.name}.1.ebwt").exists():
                bowtie_index = str(default_index)
                log.info("Auto-detected Bowtie index: %s", bowtie_index)
            else:
                log.warning(
                    "No Bowtie index specified or found at %s. "
                    "Off-target specificity columns will be NaN. "
                    "Use --bowtie-index to provide an index, or --skip-specificity to suppress this warning.",
                    default_index,
                )
    else:
        log.info("Skipping Bowtie specificity analysis (--skip-specificity flag set).")

    feature_config = FeatureConfig(
        input_csv=str(candidate_output_path),
        output_csv=str(features_output_path),
        bowtie_index=bowtie_index,
        max_mismatches=args.max_mismatches,
        threads=args.threads,
    )

    features_df = run_feature_pipeline(feature_config, gene_name=gene_name)

    # ===================================================================
    # Step 4.5: DeepSpCas9 On-Target Efficiency Prediction
    # ===================================================================
    model_metadata_path = default_out_dir / "model_metadata.json"
    deepcas9_stats = {}

    if not args.skip_deepcas9:
        log.info("")
        log.info("=" * 72)
        log.info("DEEPSPCAS9 ON-TARGET EFFICIENCY PREDICTION")
        log.info("=" * 72)
        try:
            predictor = DeepSpCas9Predictor(weights_path=args.deepcas9_weights)
            fasta_sequence = read_fasta_sequence(str(fasta_path))

            scores, deepcas9_stats = predictor.score_candidates_dataframe(
                features_df, fasta_sequence
            )
            features_df["deep_spcas9_score"] = scores

            # Save updated feature table with DeepSpCas9 scores
            features_df.to_csv(features_output_path, index=False)
            log.info(
                "DeepSpCas9 prediction complete: %d/%d predicted (failed edge cases: %d)",
                deepcas9_stats["successfully_predicted"],
                deepcas9_stats["total_candidates"],
                deepcas9_stats["failed_edge_cases"],
            )
            if deepcas9_stats["successfully_predicted"] > 0:
                log.info(
                    "Score statistics | Min: %.4f | Max: %.4f | Mean: %.4f | Median: %.4f | Std: %.4f",
                    deepcas9_stats["min"],
                    deepcas9_stats["max"],
                    deepcas9_stats["mean"],
                    deepcas9_stats["median"],
                    deepcas9_stats["std"],
                )
            log.info("Updated feature table saved to %s", features_output_path)

            # Export model metadata
            model_metadata = {
                "model_name": "DeepSpCas9",
                "paper": "Kim et al., Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity, Nature Biotechnology 2018 / Science Advances 2019",
                "source_repository": "https://github.com/MyungjaeSong/Paired-Library (branch DeepCRISPR.info)",
                "pretrained_checkpoint": "PreTrain-Final-False-3-5-7-100-70-40-0.001-550-True-80-60",
                "weights_file": str(predictor.weights_path),
                "framework": "PyTorch (converted from TensorFlow 1.x reference checkpoint)",
                "input_length_bp": 30,
                "input_format": "4 bp upstream + 20 bp guide + 3 bp PAM (NGG) + 3 bp downstream",
                "nucleotide_encoding": "One-hot [A=(1,0,0,0), C=(0,1,0,0), G=(0,0,1,0), T=(0,0,0,1)] -> [N, 4, 1, 30]",
                "output_interpretation": "Predicted on-target activity score (continuous indel frequency percentage from published high-throughput library)",
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction_statistics": deepcas9_stats,
            }
            try:
                import torch
                model_metadata["pytorch_version"] = torch.__version__
            except Exception:
                pass

            with open(model_metadata_path, "w", encoding="utf-8") as f:
                json.dump(model_metadata, f, indent=2)
            log.info("Saved model metadata JSON: %s", model_metadata_path)

        except Exception as err:
            log.error("DeepSpCas9 prediction failed: %s", err)
            features_df["deep_spcas9_score"] = float("nan")
            features_df.to_csv(features_output_path, index=False)
    else:
        log.info("Skipping DeepSpCas9 prediction (--skip-deepcas9 flag set).")
        features_df["deep_spcas9_score"] = float("nan")
        features_df.to_csv(features_output_path, index=False)

    # ===================================================================
    # Step 5: Reproducibility Record & Run Summary Export
    # ===================================================================
    run_timestamp = datetime.now(timezone.utc).isoformat()

    summary_data = {
        "gene": gene_name or "(direct FASTA)",
        "timestamp": run_timestamp,
        "ncbi_metadata": {
            "accession": seq_meta.genomic_accession if seq_meta else None,
            "assembly": seq_meta.assembly if seq_meta else None,
            "organism": seq_meta.organism if seq_meta else None,
            "ncbi_gene_id": seq_meta.ncbi_gene_id if seq_meta else None,
            "chromosome": seq_meta.chromosome if seq_meta else None,
            "strand": seq_meta.strand if seq_meta else None,
            "gene_start": seq_meta.gene_start if seq_meta else None,
            "gene_stop": seq_meta.gene_stop if seq_meta else None,
            "retrieved_start": seq_meta.retrieved_start if seq_meta else None,
            "retrieved_stop": seq_meta.retrieved_stop if seq_meta else None,
            "sequence_length": seq_meta.sequence_length if seq_meta else None,
            "fasta_path": str(fasta_path),
        },
        "candidate_counts": {
            "raw_candidates": len(candidates_df),
            "retained_candidates": len(features_df),
            "valid_20nt": int((features_df["guide_length"] == 20).sum()) if "guide_length" in features_df.columns else len(features_df),
            "invalid_length": int((features_df["guide_length"] != 20).sum()) if "guide_length" in features_df.columns else 0,
            "duplicates": int(features_df["guide_rna"].duplicated().sum()),
            "plus_strand": int((features_df["strand"] == "+").sum()),
            "minus_strand": int((features_df["strand"] == "-").sum()),
        },
        "deep_spcas9_summary": deepcas9_stats,
        "files": {
            "candidate_gRNAs_csv": str(candidate_output_path),
            "gRNA_features_csv": str(features_output_path),
            "bowtie_index": bowtie_index or "(none)",
            "model_metadata_json": str(model_metadata_path) if model_metadata_path.exists() else "(not generated)",
        },
        "feature_statistics": {},
        "missing_values": {col: int(cnt) for col, cnt in features_df.isnull().sum().items() if cnt > 0},
        "software_versions": {},
    }

    # Populate stats
    numeric_cols = [
        "gc_content", "melting_temp_c", "delta_g_kcal_mol", "folding_energy_kcal_mol",
        "doench2016_proxy_score", "paired_fraction", "off_target_count", "mismatch_score",
        "off_targets_0mm", "off_targets_1mm", "off_targets_2mm", "off_targets_3mm",
        "deep_spcas9_score"
    ]
    for col in numeric_cols:
        if col in features_df.columns and features_df[col].notna().any():
            s = features_df[col].dropna()
            summary_data["feature_statistics"][col] = {
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
                "missing": int(features_df[col].isnull().sum()),
            }

    # Software versions
    try:
        import Bio
        summary_data["software_versions"]["biopython"] = Bio.__version__
    except Exception:
        pass
    try:
        import RNA
        summary_data["software_versions"]["viennarna"] = RNA.__version__
    except Exception:
        pass
    try:
        import pandas
        summary_data["software_versions"]["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import numpy
        summary_data["software_versions"]["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import torch
        summary_data["software_versions"]["torch"] = torch.__version__
    except Exception:
        pass

    # Save summary JSON
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    log.info("Saved run summary JSON: %s", summary_json_path)

    log.info("")
    log.info("=" * 72)
    log.info("REPRODUCIBILITY RECORD")
    log.info("-" * 72)
    log.info("  Gene:              %s", gene_name or "(direct FASTA)")
    if seq_meta:
        log.info("  Accession:         %s", seq_meta.genomic_accession)
        log.info("  Assembly:          %s", seq_meta.assembly)
        log.info("  Sequence length:   %d bp", seq_meta.sequence_length)
    log.info("  Candidate count:   %d", len(features_df))
    log.info("  Feature columns:   %d", features_df.shape[1])
    log.info("  Candidates CSV:    %s", candidate_output_path)
    log.info("  Features CSV:      %s", features_output_path)
    log.info("  Summary JSON:      %s", summary_json_path)
    if model_metadata_path.exists():
        log.info("  Model Meta JSON:   %s", model_metadata_path)
    log.info("  Bowtie index:      %s", bowtie_index or "(not used)")
    log.info("  Timestamp:         %s", run_timestamp)
    for pkg, ver in summary_data["software_versions"].items():
        log.info("  %-18s %s", f"{pkg.capitalize()}:", ver)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
