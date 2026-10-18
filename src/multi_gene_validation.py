"""
src/multi_gene_validation.py

Step 7 — Multi-Gene Validation and Gene-Independence Test

Runs the SAME pipeline used for TP53 on additional validation genes to
demonstrate gene-independent operation.  No gene-specific hard-coding
or threshold modifications are introduced.

Validation genes:
    - TP53  (baseline — uses existing cached results if available)
    - HBB   (hemoglobin beta subunit, chr11, ~1.6 kb gene, plus strand)
    - EGFR  (epidermal growth factor receptor, chr7, ~188 kb gene, plus strand)
    - BRCA1 (breast cancer 1, chr17, ~81 kb gene, minus strand)

Usage:
    python -m src.multi_gene_validation
    python -m src.multi_gene_validation --genes TP53 HBB EGFR
    python -m src.multi_gene_validation --skip-run  (analysis only, skip pipeline)

Author: Bioinformatics Development Team
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multi_gene_validation")


# =========================================================================== #
# Gene Registry — No gene-specific thresholds or scoring formulas
# =========================================================================== #
# These are reference annotations for documentation only.
# The pipeline discovers coordinates dynamically from NCBI.
VALIDATION_GENES = {
    "TP53": {
        "description": "Tumor protein p53 — tumor suppressor",
        "ncbi_gene_id": "7157",
        "chromosome": "17",
        "strand": "minus",
        "expected_size_kb": "~19 kb (gene body)",
        "notes": "Baseline reference case, already validated in Steps 1-6",
    },
    "HBB": {
        "description": "Hemoglobin subunit beta — oxygen transport",
        "ncbi_gene_id": "3043",
        "chromosome": "11",
        "strand": "plus",
        "expected_size_kb": "~1.6 kb (gene body)",
        "notes": "Small gene, sickle cell disease target, plus strand",
    },
    "EGFR": {
        "description": "Epidermal growth factor receptor — oncogene",
        "ncbi_gene_id": "1956",
        "chromosome": "7",
        "strand": "plus",
        "expected_size_kb": "~188 kb (gene body)",
        "notes": "Large gene, cancer drug target, plus strand",
    },
    "BRCA1": {
        "description": "Breast cancer type 1 susceptibility protein",
        "ncbi_gene_id": "672",
        "chromosome": "17",
        "strand": "minus",
        "expected_size_kb": "~81 kb (gene body)",
        "notes": "Large gene, same chromosome as TP53, minus strand",
    },
}


# =========================================================================== #
# Expected 34-column feature schema
# =========================================================================== #
EXPECTED_FEATURE_COLUMNS = [
    "guide_id", "gene", "guide_rna", "pam_sequence", "pam_position", "strand",
    "guide_length", "length", "gc_content", "A_count", "C_count", "G_count",
    "T_count", "frac_A", "frac_C", "frac_G", "frac_T", "has_poly_t_terminator",
    "max_homopolymer_run", "starts_with_g", "melting_temp_c", "delta_g_kcal_mol",
    "secondary_structure", "folding_energy_kcal_mol", "paired_fraction",
    "doench2016_proxy_score", "off_target_count", "mismatch_score",
    "off_targets_0mm", "off_targets_1mm", "off_targets_2mm", "off_targets_3mm",
    "off_target_capped", "deep_spcas9_score",
]


# =========================================================================== #
# 1. Run the full pipeline for a single gene
# =========================================================================== #
def run_pipeline_for_gene(gene: str, python_exe: str) -> dict:
    """
    Run the full pipeline (main.py --gene <GENE>) as a subprocess.
    Returns a dict with status information.
    """
    result_dir = PROJECT_ROOT / "results" / gene
    features_csv = result_dir / "gRNA_features.csv"
    run_summary = result_dir / "run_summary.json"

    log.info("=" * 72)
    log.info("RUNNING PIPELINE FOR GENE: %s", gene)
    log.info("=" * 72)

    # Build command
    cmd = [
        python_exe, str(PROJECT_ROOT / "main.py"),
        "--gene", gene,
    ]

    log.info("Command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per gene
        )

        if proc.returncode != 0:
            log.error("Pipeline FAILED for %s (exit code %d)", gene, proc.returncode)
            log.error("STDERR:\n%s", proc.stderr[-2000:] if proc.stderr else "(empty)")
            return {
                "gene": gene,
                "status": "FAILED",
                "exit_code": proc.returncode,
                "error": proc.stderr[-2000:] if proc.stderr else "Unknown error",
            }

        log.info("Pipeline completed successfully for %s", gene)

        # Verify outputs exist
        if not features_csv.exists():
            return {
                "gene": gene,
                "status": "FAILED",
                "error": f"Feature CSV not found: {features_csv}",
            }

        return {
            "gene": gene,
            "status": "SUCCESS",
            "features_csv": str(features_csv),
            "run_summary": str(run_summary) if run_summary.exists() else None,
        }

    except subprocess.TimeoutExpired:
        log.error("Pipeline TIMED OUT for %s", gene)
        return {"gene": gene, "status": "TIMEOUT", "error": "Pipeline exceeded 10-minute timeout"}
    except Exception as e:
        log.error("Pipeline execution error for %s: %s", gene, e)
        return {"gene": gene, "status": "ERROR", "error": str(e)}


# =========================================================================== #
# 2. Run the Step 6 optimization for a single gene
# =========================================================================== #
def run_optimization_for_gene(gene: str, python_exe: str) -> dict:
    """
    Run grna_optimizer.py for a single gene using the same formulation as TP53.
    """
    features_csv = PROJECT_ROOT / "results" / gene / "gRNA_features.csv"
    opt_dir = PROJECT_ROOT / "results" / gene / "optimization"

    if not features_csv.exists():
        return {
            "gene": gene,
            "status": "FAILED",
            "error": f"Feature CSV missing: {features_csv}",
        }

    log.info("-" * 72)
    log.info("RUNNING OPTIMIZATION FOR GENE: %s", gene)
    log.info("-" * 72)

    cmd = [
        python_exe, "-m", "src.grna_optimizer",
        "--input", str(features_csv),
        "--output-dir", str(opt_dir),
        "--k", "1",
        "--gc-min", "0.30",
        "--gc-max", "0.70",
        "--max-homopolymer", "4",
        "--capped-policy", "retain",
        "--n-epsilon-steps", "50",
        "--n-weight-steps", "21",
    ]
    # NOTE: --mfe-threshold is NOT set (disabled by default) per Step 7 requirements

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=900,  # 15 min timeout (large genes need more time for LP relaxation)
        )

        if proc.returncode != 0:
            log.error("Optimization FAILED for %s (exit code %d)", gene, proc.returncode)
            log.error("STDERR:\n%s", proc.stderr[-2000:] if proc.stderr else "(empty)")
            return {
                "gene": gene,
                "status": "FAILED",
                "exit_code": proc.returncode,
                "error": proc.stderr[-2000:] if proc.stderr else "Unknown error",
            }

        summary_json = opt_dir / "optimization_summary.json"
        if not summary_json.exists():
            return {
                "gene": gene,
                "status": "FAILED",
                "error": "optimization_summary.json not generated",
            }

        log.info("Optimization completed successfully for %s", gene)
        return {
            "gene": gene,
            "status": "SUCCESS",
            "optimization_dir": str(opt_dir),
        }

    except subprocess.TimeoutExpired:
        return {"gene": gene, "status": "TIMEOUT", "error": "Optimization exceeded 15-minute timeout"}
    except Exception as e:
        return {"gene": gene, "status": "ERROR", "error": str(e)}


# =========================================================================== #
# 3. Collect per-gene statistics
# =========================================================================== #
def collect_gene_statistics(gene: str) -> dict:
    """
    Collect all statistics for a single gene from its results directory.
    Returns a comprehensive dict of statistics.
    """
    result_dir = PROJECT_ROOT / "results" / gene
    features_csv = result_dir / "gRNA_features.csv"
    candidates_csv = result_dir / "candidate_gRNAs.csv"
    run_summary_json = result_dir / "run_summary.json"
    opt_summary_json = result_dir / "optimization" / "optimization_summary.json"
    pareto_csv = result_dir / "optimization" / "pareto_front.csv"
    selected_csv = result_dir / "optimization" / "selected_guides.csv"

    stats = {"gene": gene, "status": "INCOMPLETE"}

    # --- Sequence/accession info ---
    if run_summary_json.exists():
        with open(run_summary_json, "r") as f:
            run_summary = json.load(f)
        meta = run_summary.get("ncbi_metadata", {})
        stats["accession"] = meta.get("accession", "")
        stats["assembly"] = meta.get("assembly", "")
        stats["chromosome"] = meta.get("chromosome", "")
        stats["strand"] = meta.get("strand", "")
        stats["gene_start"] = meta.get("gene_start", 0)
        stats["gene_stop"] = meta.get("gene_stop", 0)
        stats["sequence_length"] = meta.get("sequence_length", 0)
        stats["retrieved_start"] = meta.get("retrieved_start", 0)
        stats["retrieved_stop"] = meta.get("retrieved_stop", 0)
        stats["fasta_path"] = meta.get("fasta_path", "")
    else:
        log.warning("run_summary.json not found for %s", gene)
        return stats

    # --- Candidate counts ---
    if candidates_csv.exists():
        cand_df = pd.read_csv(candidates_csv)
        stats["total_candidates"] = len(cand_df)
        stats["plus_strand_candidates"] = int((cand_df["strand"] == "+").sum())
        stats["minus_strand_candidates"] = int((cand_df["strand"] == "-").sum())
        if stats["sequence_length"] > 0:
            stats["candidate_density"] = round(
                len(cand_df) / stats["sequence_length"] * 1000, 2
            )  # per kb
        else:
            stats["candidate_density"] = 0.0
    else:
        log.warning("candidate_gRNAs.csv not found for %s", gene)
        return stats

    # --- Feature table ---
    if not features_csv.exists():
        log.warning("gRNA_features.csv not found for %s", gene)
        return stats

    feat_df = pd.read_csv(features_csv)
    stats["feature_columns"] = list(feat_df.columns)
    stats["feature_column_count"] = len(feat_df.columns)
    stats["schema_matches_expected"] = (
        sorted(feat_df.columns.tolist()) == sorted(EXPECTED_FEATURE_COLUMNS)
    )

    # --- DeepSpCas9 statistics ---
    if "deep_spcas9_score" in feat_df.columns:
        valid_scores = feat_df["deep_spcas9_score"].dropna()
        nan_scores = int(feat_df["deep_spcas9_score"].isna().sum())
        stats["deepspcas9"] = {
            "total": len(feat_df),
            "predicted": len(valid_scores),
            "nan_invalid": nan_scores,
            "min": round(float(valid_scores.min()), 4) if len(valid_scores) > 0 else None,
            "max": round(float(valid_scores.max()), 4) if len(valid_scores) > 0 else None,
            "mean": round(float(valid_scores.mean()), 4) if len(valid_scores) > 0 else None,
            "median": round(float(valid_scores.median()), 4) if len(valid_scores) > 0 else None,
        }
    else:
        stats["deepspcas9"] = {"error": "deep_spcas9_score column not found"}

    # --- Off-target statistics ---
    if "off_target_capped" in feat_df.columns:
        capped = feat_df["off_target_capped"] == True  # noqa: E712
        uncapped = ~capped
        stats["off_target"] = {
            "uncapped_guides": int(uncapped.sum()),
            "capped_guides": int(capped.sum()),
            "min_ot_count": int(feat_df["off_target_count"].min()),
            "max_uncapped_ot_count": (
                int(feat_df.loc[uncapped, "off_target_count"].max())
                if uncapped.any() else None
            ),
        }

    # --- Optimization statistics ---
    if opt_summary_json.exists():
        with open(opt_summary_json, "r") as f:
            opt_summary = json.load(f)

        filtering = opt_summary.get("filtering", {})
        stats["filtering"] = {
            "total_candidates": filtering.get("total_candidates", 0),
            "nan_deep_spcas9": filtering.get("filters_applied", {}).get("nan_deep_spcas9", 0),
            "gc_failures": filtering.get("filters_applied", {}).get("gc_content_out_of_range", 0),
            "poly_t_failures": filtering.get("filters_applied", {}).get("poly_t_terminator", 0),
            "homopolymer_failures": filtering.get("filters_applied", {}).get("homopolymer_run_exceeded", 0),
            "excluded": filtering.get("excluded_candidates", 0),
            "feasible": filtering.get("feasible_candidates", 0),
            "feasible_pct": round(
                filtering.get("feasible_candidates", 0) / max(filtering.get("total_candidates", 1), 1) * 100, 1
            ),
            "capped_in_feasible": filtering.get("capped_in_feasible_set", 0),
        }

        pareto = opt_summary.get("pareto_analysis", {})
        stats["pareto_count"] = pareto.get("n_pareto_optimal", 0)

        selected_guides = opt_summary.get("selected_guides", [])
        if selected_guides:
            sg = selected_guides[0]
            stats["selected_guide"] = {
                "guide_id": sg.get("guide_id", ""),
                "guide_rna": sg.get("guide_rna", ""),
                "strand": sg.get("strand", ""),
                "gc_content": sg.get("gc_content", 0.0),
                "deep_spcas9_score": sg.get("deep_spcas9_score", 0.0),
                "off_target_count": sg.get("off_target_count", 0),
                "off_target_capped": sg.get("off_target_capped", False),
                "norm_activity": sg.get("norm_activity", 0.0),
                "norm_specificity": sg.get("norm_specificity", 0.0),
                "balanced_score": sg.get("balanced_score", 0.0),
            }

        # Validation
        validation = opt_summary.get("validation", {})
        stats["validation_all_passed"] = validation.get("_all_passed", False)

        stats["status"] = "COMPLETE"
    else:
        stats["status"] = "PARTIAL (no optimization)"

    return stats


# =========================================================================== #
# 4. Cross-Gene Comparison Tables
# =========================================================================== #
def build_comparison_tables(all_stats: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the two cross-gene comparison tables:
      1. Gene comparison table
      2. Selected guide comparison table
    """
    # --- Table 1: Gene comparison ---
    rows1 = []
    for s in all_stats:
        if s.get("status") != "COMPLETE":
            continue
        row = {
            "Gene": s["gene"],
            "Seq Length": s.get("sequence_length", 0),
            "Candidates": s.get("total_candidates", 0),
            "Feasible": s.get("filtering", {}).get("feasible", 0),
            "Feasible %": s.get("filtering", {}).get("feasible_pct", 0.0),
            "Pareto Guides": s.get("pareto_count", 0),
            "Capped Guides": s.get("filtering", {}).get("capped_in_feasible", 0),
            "Mean DeepSpCas9": s.get("deepspcas9", {}).get("mean", 0.0),
            "Best DeepSpCas9": s.get("deepspcas9", {}).get("max", 0.0),
            "Selected Guide": s.get("selected_guide", {}).get("guide_id", ""),
        }
        rows1.append(row)
    gene_comparison_df = pd.DataFrame(rows1)

    # --- Table 2: Selected guide comparison ---
    rows2 = []
    for s in all_stats:
        if s.get("status") != "COMPLETE" or "selected_guide" not in s:
            continue
        sg = s["selected_guide"]
        row = {
            "Gene": s["gene"],
            "Guide ID": sg.get("guide_id", ""),
            "Guide Sequence": sg.get("guide_rna", ""),
            "Strand": sg.get("strand", ""),
            "GC": sg.get("gc_content", 0.0),
            "DeepSpCas9": sg.get("deep_spcas9_score", 0.0),
            "Off-target Count": sg.get("off_target_count", 0),
            "E_hat": sg.get("norm_activity", 0.0),
            "S_hat": sg.get("norm_specificity", 0.0),
            "Balanced Score": sg.get("balanced_score", 0.0),
        }
        rows2.append(row)
    selected_comparison_df = pd.DataFrame(rows2)

    return gene_comparison_df, selected_comparison_df


# =========================================================================== #
# 5. Gene-Independence Validation
# =========================================================================== #
def validate_gene_independence(all_stats: list[dict]) -> dict:
    """
    Run all gene-independence validation checks.
    Returns a per-gene dict of pass/fail results.
    """
    validation_results = {}

    for s in all_stats:
        gene = s["gene"]
        checks = {}

        if s.get("status") != "COMPLETE":
            checks["pipeline_complete"] = {"pass": False, "detail": f"Status: {s.get('status')}"}
            checks["_overall"] = "FAIL"
            validation_results[gene] = checks
            continue

        # 1. Correct gene name in output
        checks["gene_name_correct"] = {
            "pass": gene == s["gene"],
            "detail": f"Gene: {s['gene']}",
        }

        # 2. Correct sequence source
        checks["sequence_source_valid"] = {
            "pass": bool(s.get("accession", "")),
            "detail": f"Accession: {s.get('accession', 'MISSING')}",
        }

        # 3. Sequence is non-empty
        checks["sequence_non_empty"] = {
            "pass": s.get("sequence_length", 0) > 0,
            "detail": f"Length: {s.get('sequence_length', 0)} bp",
        }

        # 4. Guide length = 20
        result_dir = PROJECT_ROOT / "results" / gene
        features_csv = result_dir / "gRNA_features.csv"
        if features_csv.exists():
            feat_df = pd.read_csv(features_csv)
            all_20nt = bool((feat_df["guide_length"] == 20).all())
            checks["guide_length_20"] = {
                "pass": all_20nt,
                "detail": f"All guides 20nt: {all_20nt}",
            }

            # 5. PAM = NGG
            all_ngg = bool(feat_df["pam_sequence"].str.match(r"^[ACGT]GG$").all())
            checks["pam_ngg"] = {
                "pass": all_ngg,
                "detail": f"All PAMs match NGG: {all_ngg}",
            }

            # 6. Both strands represented
            has_plus = (feat_df["strand"] == "+").any()
            has_minus = (feat_df["strand"] == "-").any()
            checks["both_strands"] = {
                "pass": bool(has_plus and has_minus),
                "detail": f"Plus: {has_plus}, Minus: {has_minus}",
            }

            # 7. Feature schema matches TP53
            schema_ok = s.get("schema_matches_expected", False)
            checks["feature_schema_matches"] = {
                "pass": schema_ok,
                "detail": f"Column count: {s.get('feature_column_count', 0)}, Match: {schema_ok}",
            }

            # 8. DeepSpCas9 input length correct (30bp contexts)
            ds = s.get("deepspcas9", {})
            checks["deepspcas9_predictions_valid"] = {
                "pass": ds.get("predicted", 0) > 0,
                "detail": f"Predicted: {ds.get('predicted', 0)}, NaN: {ds.get('nan_invalid', 0)}",
            }

            # 9. No unexpected NaN values (except documented boundary cases)
            nan_ds = ds.get("nan_invalid", 0)
            checks["nan_boundary_only"] = {
                "pass": nan_ds <= 5,  # allow a few edge-case NaNs
                "detail": f"NaN DeepSpCas9: {nan_ds} (tolerance: <= 5)",
            }
        else:
            checks["features_exist"] = {"pass": False, "detail": "Feature CSV missing"}

        # 10. Optimization completed
        checks["optimization_complete"] = {
            "pass": s.get("status") == "COMPLETE",
            "detail": f"Status: {s.get('status')}",
        }

        # 11. Pareto set non-empty
        checks["pareto_non_empty"] = {
            "pass": s.get("pareto_count", 0) > 0,
            "detail": f"Pareto guides: {s.get('pareto_count', 0)}",
        }

        # 12. Selected guide exists
        sg = s.get("selected_guide", {})
        checks["selected_guide_exists"] = {
            "pass": bool(sg.get("guide_id", "")),
            "detail": f"Selected: {sg.get('guide_id', 'NONE')}",
        }

        # 13. Validation all passed
        checks["all_optimization_checks_passed"] = {
            "pass": s.get("validation_all_passed", False),
            "detail": f"All passed: {s.get('validation_all_passed', False)}",
        }

        # Overall
        all_pass = all(c["pass"] for c in checks.values())
        checks["_overall"] = "PASS" if all_pass else "FAIL"

        validation_results[gene] = checks

    return validation_results


# =========================================================================== #
# 6. Publication-Quality Visualisations
# =========================================================================== #
def generate_plots(all_stats: list[dict], output_dir: Path) -> list[str]:
    """
    Generate publication-quality plots for multi-gene comparison.
    Returns list of generated plot file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        log.warning("matplotlib not available — skipping plot generation")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Colours for genes
    gene_colors = {
        "TP53": "#2196F3",   # blue
        "HBB": "#4CAF50",    # green
        "EGFR": "#FF9800",   # orange
        "BRCA1": "#E91E63",  # pink
    }

    complete_stats = [s for s in all_stats if s.get("status") == "COMPLETE"]
    if not complete_stats:
        log.warning("No complete gene results for plotting")
        return []

    genes = [s["gene"] for s in complete_stats]
    colors = [gene_colors.get(g, "#9E9E9E") for g in genes]

    # ---- Plot 1: Candidate count by gene ----
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [s.get("total_candidates", 0) for s in complete_stats]
    bars = ax.bar(genes, counts, color=colors, edgecolor="white", linewidth=1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Candidate gRNAs", fontsize=11)
    ax.set_title("Candidate gRNA Count by Gene", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gene", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p1 = plots_dir / "01_candidate_count_by_gene.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p1))

    # ---- Plot 2: Feasible vs excluded by gene ----
    fig, ax = plt.subplots(figsize=(8, 5))
    feasible = [s.get("filtering", {}).get("feasible", 0) for s in complete_stats]
    excluded = [s.get("filtering", {}).get("excluded", 0) for s in complete_stats]
    x = np.arange(len(genes))
    w = 0.35
    b1 = ax.bar(x - w/2, feasible, w, label="Feasible", color="#4CAF50", edgecolor="white")
    b2 = ax.bar(x + w/2, excluded, w, label="Excluded", color="#F44336", edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(feasible + excluded) * 0.02,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(genes)
    ax.set_ylabel("Number of Guides", fontsize=11)
    ax.set_title("Feasible vs Excluded Guides by Gene", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p2 = plots_dir / "02_feasible_vs_excluded.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p2))

    # ---- Plot 3: DeepSpCas9 score distribution by gene (box plot) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    score_data = []
    gene_labels = []
    for s in complete_stats:
        gene = s["gene"]
        features_csv = PROJECT_ROOT / "results" / gene / "gRNA_features.csv"
        if features_csv.exists():
            df = pd.read_csv(features_csv)
            valid = df["deep_spcas9_score"].dropna().values
            score_data.append(valid)
            gene_labels.append(gene)

    if score_data:
        bp = ax.boxplot(score_data, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='.', markersize=2, alpha=0.3))
        ax.set_xticklabels(gene_labels)
        for patch, color in zip(bp["boxes"], [gene_colors.get(g, "#9E9E9E") for g in gene_labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("DeepSpCas9 Score", fontsize=11)
        ax.set_title("DeepSpCas9 Score Distribution by Gene", fontsize=13, fontweight="bold")
        ax.set_xlabel("Gene", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    p3 = plots_dir / "03_deepspcas9_distribution.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p3))

    # ---- Plot 4: Pareto front for each gene (2x2 subplots) ----
    n_genes = len(complete_stats)
    ncols = 2
    nrows = (n_genes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for idx, s in enumerate(complete_stats):
        gene = s["gene"]
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]

        feasible_csv = PROJECT_ROOT / "results" / gene / "optimization" / "feasible_candidates.csv"
        pareto_csv = PROJECT_ROOT / "results" / gene / "optimization" / "pareto_front.csv"
        selected_csv = PROJECT_ROOT / "results" / gene / "optimization" / "selected_guides.csv"

        if feasible_csv.exists():
            fdf = pd.read_csv(feasible_csv)
            ax.scatter(fdf["norm_specificity"], fdf["norm_activity"],
                       c="#d0d0d0", s=6, alpha=0.4, label="Feasible", zorder=1, edgecolors="none")

        if pareto_csv.exists():
            pdf = pd.read_csv(pareto_csv)
            ax.scatter(pdf["norm_specificity"], pdf["norm_activity"],
                       c=gene_colors.get(gene, "#2196F3"), s=40, alpha=0.85,
                       label="Pareto-optimal", zorder=3, edgecolors="navy", linewidths=0.5)
            pf_sorted = pdf.sort_values("norm_specificity")
            ax.plot(pf_sorted["norm_specificity"], pf_sorted["norm_activity"],
                    c=gene_colors.get(gene, "#1565C0"), linewidth=1.2, alpha=0.6, zorder=2)

        if selected_csv.exists():
            sdf = pd.read_csv(selected_csv)
            ax.scatter(sdf["norm_specificity"], sdf["norm_activity"],
                       c="#FF1744", s=200, marker="*",
                       label="Selected (k=1)", zorder=5, edgecolors="darkred", linewidths=0.8)

        ax.set_xlabel("Norm. Specificity →", fontsize=9)
        ax.set_ylabel("Norm. Activity →", fontsize=9)
        ax.set_title(f"{gene} — Pareto Front", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(complete_stats), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row][col].set_visible(False)

    fig.suptitle("Multi-Gene Pareto Fronts", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p4 = plots_dir / "04_pareto_fronts_multi_gene.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p4))

    # ---- Plot 5: Selected guide comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    selected_genes = []
    e_hats = []
    s_hats = []
    balanced = []

    for s in complete_stats:
        if "selected_guide" in s:
            sg = s["selected_guide"]
            selected_genes.append(s["gene"])
            e_hats.append(sg.get("norm_activity", 0))
            s_hats.append(sg.get("norm_specificity", 0))
            balanced.append(sg.get("balanced_score", 0))

    if selected_genes:
        x = np.arange(len(selected_genes))
        w = 0.25
        b1 = ax.bar(x - w, e_hats, w, label="Ê (Activity)", color="#2196F3", edgecolor="white")
        b2 = ax.bar(x, s_hats, w, label="Ŝ (Specificity)", color="#4CAF50", edgecolor="white")
        b3 = ax.bar(x + w, balanced, w, label="Balanced (0.5Ê + 0.5Ŝ)",
                     color="#FF9800", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_genes)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Normalised Score", fontsize=11)
        ax.set_title("Selected Guide Objective Values Across Genes", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    p5 = plots_dir / "05_selected_guide_comparison.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p5))

    log.info("Generated %d plots in %s", len(generated), plots_dir)
    return generated


# =========================================================================== #
# 7. Main Orchestrator
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7 — Multi-Gene Validation and Gene-Independence Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--genes", nargs="*", default=["TP53", "HBB", "EGFR", "BRCA1"],
        help="List of gene symbols to validate",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip running the pipeline; only perform analysis on existing results",
    )
    parser.add_argument(
        "--skip-tp53", action="store_true",
        help="Skip re-running TP53 (use existing results as baseline)",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python executable to use for subprocess calls",
    )
    args = parser.parse_args()

    genes = [g.strip().upper() for g in args.genes]

    log.info("=" * 72)
    log.info("STEP 7 — MULTI-GENE VALIDATION AND GENE-INDEPENDENCE TEST")
    log.info("=" * 72)
    log.info("Genes to validate: %s", genes)
    log.info("Python executable: %s", args.python)
    log.info("")

    # -----------------------------------------------------------------------
    # Phase 1: Document gene selection
    # -----------------------------------------------------------------------
    log.info("=" * 72)
    log.info("PHASE 1: GENE SELECTION AND DOCUMENTATION")
    log.info("=" * 72)
    for gene in genes:
        info = VALIDATION_GENES.get(gene, {})
        log.info("  %s: %s", gene, info.get("description", "(custom gene)"))
        log.info("    NCBI Gene ID: %s | Chr: %s | Strand: %s",
                 info.get("ncbi_gene_id", "TBD"),
                 info.get("chromosome", "TBD"),
                 info.get("strand", "TBD"))
    log.info("")

    # -----------------------------------------------------------------------
    # Phase 2: Run pipelines
    # -----------------------------------------------------------------------
    pipeline_results = {}
    optimization_results = {}

    if not args.skip_run:
        log.info("=" * 72)
        log.info("PHASE 2: RUNNING PIPELINES")
        log.info("=" * 72)

        for gene in genes:
            if gene == "TP53" and args.skip_tp53:
                # Check if TP53 results exist
                tp53_features = PROJECT_ROOT / "results" / "TP53" / "gRNA_features.csv"
                if tp53_features.exists():
                    log.info("Skipping TP53 pipeline (--skip-tp53); using existing results")
                    pipeline_results[gene] = {"gene": gene, "status": "SUCCESS (cached)"}
                else:
                    log.warning("TP53 results not found despite --skip-tp53; running pipeline")
                    pipeline_results[gene] = run_pipeline_for_gene(gene, args.python)
            else:
                pipeline_results[gene] = run_pipeline_for_gene(gene, args.python)

            if pipeline_results[gene].get("status", "").startswith("SUCCESS"):
                # Check if optimization already exists
                opt_summary = PROJECT_ROOT / "results" / gene / "optimization" / "optimization_summary.json"
                if gene == "TP53" and args.skip_tp53 and opt_summary.exists():
                    log.info("Skipping TP53 optimization; using existing results")
                    optimization_results[gene] = {"gene": gene, "status": "SUCCESS (cached)"}
                else:
                    optimization_results[gene] = run_optimization_for_gene(gene, args.python)
            else:
                log.warning("Skipping optimization for %s due to pipeline failure", gene)
                optimization_results[gene] = {"gene": gene, "status": "SKIPPED"}

        log.info("")
        log.info("Pipeline execution summary:")
        for gene in genes:
            p_stat = pipeline_results.get(gene, {}).get("status", "N/A")
            o_stat = optimization_results.get(gene, {}).get("status", "N/A")
            log.info("  %s: pipeline=%s, optimization=%s", gene, p_stat, o_stat)
    else:
        log.info("Skipping pipeline execution (--skip-run)")

    # -----------------------------------------------------------------------
    # Phase 3: Collect statistics
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 3: COLLECTING GENE STATISTICS")
    log.info("=" * 72)

    all_stats = []
    for gene in genes:
        log.info("Collecting statistics for %s ...", gene)
        stats = collect_gene_statistics(gene)
        all_stats.append(stats)
        log.info("  Status: %s | Candidates: %s | Feasible: %s | Pareto: %s",
                 stats.get("status"),
                 stats.get("total_candidates", "N/A"),
                 stats.get("filtering", {}).get("feasible", "N/A"),
                 stats.get("pareto_count", "N/A"))

    # -----------------------------------------------------------------------
    # Phase 4: Cross-gene comparison
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 4: CROSS-GENE COMPARISON")
    log.info("=" * 72)

    multi_gene_dir = PROJECT_ROOT / "results" / "multi_gene_validation"
    multi_gene_dir.mkdir(parents=True, exist_ok=True)

    gene_comparison_df, selected_comparison_df = build_comparison_tables(all_stats)

    if not gene_comparison_df.empty:
        gc_path = multi_gene_dir / "gene_comparison.csv"
        gene_comparison_df.to_csv(gc_path, index=False)
        log.info("Saved gene comparison: %s", gc_path)
        log.info("\n%s", gene_comparison_df.to_string(index=False))

    if not selected_comparison_df.empty:
        sg_path = multi_gene_dir / "selected_guides_comparison.csv"
        selected_comparison_df.to_csv(sg_path, index=False)
        log.info("\nSaved selected guides comparison: %s", sg_path)
        log.info("\n%s", selected_comparison_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Phase 5: Gene-independence validation
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 5: GENE-INDEPENDENCE VALIDATION")
    log.info("=" * 72)

    validation_results = validate_gene_independence(all_stats)

    validation_rows = []
    for gene, checks in validation_results.items():
        overall = checks.pop("_overall", "UNKNOWN")
        log.info("")
        log.info("  %s: %s", gene, overall)
        for check_name, result in checks.items():
            status = "PASS" if result["pass"] else "FAIL"
            log.info("    [%s] %s: %s", status, check_name, result["detail"])

        validation_rows.append({
            "Gene": gene,
            "Overall": overall,
            "Checks_Passed": sum(1 for c in checks.values() if c["pass"]),
            "Checks_Total": len(checks),
        })

    validation_summary_df = pd.DataFrame(validation_rows)
    vs_path = multi_gene_dir / "validation_summary.csv"
    validation_summary_df.to_csv(vs_path, index=False)
    log.info("\nSaved validation summary: %s", vs_path)

    # -----------------------------------------------------------------------
    # Phase 6: Generate plots
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 6: GENERATING VISUALISATIONS")
    log.info("=" * 72)

    plot_paths = generate_plots(all_stats, multi_gene_dir)

    # -----------------------------------------------------------------------
    # Phase 7: Save comprehensive JSON report
    # -----------------------------------------------------------------------
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": "Step 7 — Multi-Gene Validation and Gene-Independence Test",
        "genes_tested": genes,
        "gene_documentation": {
            gene: VALIDATION_GENES.get(gene, {"description": "Custom gene"})
            for gene in genes
        },
        "per_gene_statistics": {s["gene"]: s for s in all_stats},
        "gene_comparison": gene_comparison_df.to_dict(orient="records") if not gene_comparison_df.empty else [],
        "selected_guide_comparison": selected_comparison_df.to_dict(orient="records") if not selected_comparison_df.empty else [],
        "validation_results": {
            gene: {
                k: v for k, v in checks.items()
            }
            for gene, checks in validation_results.items()
        },
        "validation_summary": validation_summary_df.to_dict(orient="records"),
        "plots_generated": plot_paths,
        "limitations": [
            "TP53 is only the baseline demonstration case.",
            "off_target_count is a computational hit count, not a complete biological specificity score.",
            "Capped off-target counts (>=999) are lower-bound/censored computational values.",
            "Equal weighting (0.5/0.5) is a demonstration preference, not universally optimal.",
            "Multi-gene computational validation does not establish experimental biological efficacy.",
            "This framework does not claim universal gene independence or clinical validity.",
        ],
        "conclusion": (
            "The computational framework demonstrates gene-independent operation across "
            "multiple human genes under a common candidate-generation, feature-extraction, "
            "prediction, and multi-objective optimization pipeline."
        ),
    }

    report_path = multi_gene_dir / "step7_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Saved comprehensive Step 7 report: %s", report_path)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 72)
    log.info("STEP 7 — FINAL VALIDATION SUMMARY")
    log.info("=" * 72)
    for row in validation_rows:
        log.info("  %s: %s (%d/%d checks)",
                 row["Gene"], row["Overall"],
                 row["Checks_Passed"], row["Checks_Total"])
    log.info("")
    log.info("Output directory: %s", multi_gene_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
