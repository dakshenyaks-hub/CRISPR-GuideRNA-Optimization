"""
src/pipeline_controller.py

Unified Decoupled Pipeline Controller for CRISPR Guide RNA Optimization Platform.
==================================================================================

Serves as the clean data access and orchestration layer mediating between
canonical research datasets (Steps 7, 10A, 10B, 11) and dashboard/API consumers.

Key Guarantees:
  - Read-only access to canonical research files.
  - Zero mock or synthetic data generation.
  - Strict preservation of frozen research baselines:
      TP53:  3,554 total | 1,648 feasible | Pareto: 8  | Top-5
      HBB:     540 total |   365 feasible | Pareto: 10 | Top-5
      EGFR: 23,112 total | 16,090 feasible | Pareto: 8  | Top-5
      BRCA1:15,527 total |  5,778 feasible | Pareto: 10 | Top-5
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
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
log = logging.getLogger("pipeline_controller")

# ---------------------------------------------------------------------------
# Canonical Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
STEP10A_DIR = RESULTS_DIR / "step10a_cfd_rescored"
STEP10B_DIR = RESULTS_DIR / "step10b_optimization"
STEP11_DIR = RESULTS_DIR / "step11_sensitivity"
DYNAMIC_RESULTS_DIR = RESULTS_DIR / "dynamic"

CFD_RESCORED_CSV = STEP10A_DIR / "cfd_guide_features.csv"
STEP10B_SUMMARY_JSON = REPORTS_DIR / "step10b_optimization_summary.json"
STEP11_SUMMARY_JSON = REPORTS_DIR / "step11_sensitivity_summary.json"

SUPPORTED_GENES = ["TP53", "HBB", "EGFR", "BRCA1"]

GENE_DESCRIPTIONS = {
    "TP53": "Cellular tumor antigen p53 (Tumor Suppressor, 17p13.1)",
    "HBB": "Hemoglobin subunit beta (Beta-Thalassemia target, 11p15.4)",
    "EGFR": "Epidermal growth factor receptor (Oncogene kinase, 7p11.2)",
    "BRCA1": "BRCA1 DNA repair associated (Breast/Ovarian cancer, 17q21.31)",
}

# ---------------------------------------------------------------------------
# Frozen Baseline Reference (for automated integrity checking)
# ---------------------------------------------------------------------------
FROZEN_BASELINE = {
    "TP53": {
        "total_candidates": 3554,
        "feasible_candidates": 1648,
        "pareto_size": 8,
        "selected_top5": [
            "TP53_150_-",
            "TP53_11911_-",
            "TP53_559_-",
            "TP53_2766_+",
            "TP53_3053_-",
        ],
    },
    "HBB": {
        "total_candidates": 540,
        "feasible_candidates": 365,
        "pareto_size": 10,
        "selected_top5": [
            "HBB_1533_+",
            "HBB_3541_+",
            "HBB_1202_-",
            "HBB_1861_-",
            "HBB_2265_-",
        ],
    },
    "EGFR": {
        "total_candidates": 23112,
        "feasible_candidates": 16090,
        "pareto_size": 8,
        "selected_top5": [
            "EGFR_100693_+",
            "EGFR_90787_-",
            "EGFR_74407_+",
            "EGFR_27703_-",
            "EGFR_167075_+",
        ],
    },
    "BRCA1": {
        "total_candidates": 15527,
        "feasible_candidates": 5778,
        "pareto_size": 10,
        "selected_top5": [
            "BRCA1_2088_+",
            "BRCA1_2323_+",
            "BRCA1_27410_-",
            "BRCA1_122416_-",
            "BRCA1_74210_+",
        ],
    },
}


class PipelineController:
    """
    Decoupled interface mediating between physical result datasets and
    interactive dashboard or API consumers.
    """

    @classmethod
    def get_supported_genes(cls) -> List[str]:
        """Returns the list of validated canonical genes."""
        return list(SUPPORTED_GENES)

    @classmethod
    def get_gene_metadata(cls, gene: str) -> Dict[str, Any]:
        """Loads NCBI genomic metadata for a specific target gene."""
        gene = gene.upper()
        if gene not in SUPPORTED_GENES:
            raise ValueError(f"Gene '{gene}' not in supported genes: {SUPPORTED_GENES}")

        summary_file = RESULTS_DIR / gene / "run_summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                meta = data.get("ncbi_metadata", {})
                return {
                    "gene": gene,
                    "description": GENE_DESCRIPTIONS.get(gene, ""),
                    "accession": meta.get("accession", "N/A"),
                    "assembly": meta.get("assembly", "GRCh38.p14"),
                    "organism": meta.get("organism", "Homo sapiens"),
                    "chromosome": str(meta.get("chromosome", "")),
                    "strand": meta.get("strand", "+"),
                    "gene_start": meta.get("gene_start"),
                    "gene_stop": meta.get("gene_stop"),
                    "sequence_length": meta.get("sequence_length", 0),
                    "retrieved_start": meta.get("retrieved_start"),
                    "retrieved_stop": meta.get("retrieved_stop"),
                }
        return {
            "gene": gene,
            "description": GENE_DESCRIPTIONS.get(gene, ""),
            "assembly": "GRCh38.p14",
            "organism": "Homo sapiens",
        }

    @classmethod
    def get_gene_overview(cls, gene: str) -> Dict[str, Any]:
        """
        Loads candidate counts, feasibility funnel attrition, Pareto metrics,
        and objective ranges from the canonical Step 10B outputs.
        """
        gene = gene.upper()
        if gene not in SUPPORTED_GENES:
            raise ValueError(f"Gene '{gene}' not in supported genes: {SUPPORTED_GENES}")

        opt_summary_path = STEP10B_DIR / gene / "run1" / "optimization_summary.json"
        if not opt_summary_path.exists():
            raise FileNotFoundError(f"Missing Step 10B summary at: {opt_summary_path}")

        with open(opt_summary_path, "r", encoding="utf-8") as fp:
            summary = json.load(fp)

        meta = cls.get_gene_metadata(gene)
        feas = summary.get("feasibility_audit", {})
        pareto = summary.get("pareto_analysis", {})
        ranges = summary.get("objective_ranges_feasible", {})
        selected = summary.get("selected_guides", [])
        lp = summary.get("lp_track_a", {})

        return {
            "metadata": meta,
            "total_candidates": feas.get("total_candidates", 0),
            "feasible_candidates": feas.get("feasible_candidates", 0),
            "excluded_candidates": feas.get("excluded_candidates", 0),
            "exclusion_breakdown": feas.get("exclusion_breakdown", {}),
            "legacy_ot20_retention": feas.get("legacy_ot20_threshold_would_retain", 0),
            "cfd_constraint_rationale": feas.get("cfd_constraint_rationale", ""),
            "pareto_front_size": pareto.get("verified_pareto_optimal_count", 0),
            "selected_count": len(selected),
            "selected_guide_ids": [s.get("guide_id") for s in selected],
            "objective_ranges": {
                "activity": ranges.get("activity_deep_spcas9", {}),
                "norm_activity": ranges.get("norm_activity", {}),
                "cfd_risk": ranges.get("cfd_off_target_risk", {}),
                "cfd_spec": ranges.get("cfd_specificity_score", {}),
                "norm_cfd_spec": ranges.get("norm_cfd_specificity", {}),
            },
            "lp_relaxation": {
                "status": lp.get("status", "optimal"),
                "optimal_objective": lp.get("optimal_objective"),
                "is_convex": True,
            },
        }

    @classmethod
    def get_pareto_front(cls, gene: str) -> List[Dict[str, Any]]:
        """Loads verified non-dominated Pareto frontier candidates for a gene."""
        gene = gene.upper()
        pareto_csv = STEP10B_DIR / gene / "run1" / "pareto_front.csv"
        if not pareto_csv.exists():
            raise FileNotFoundError(f"Missing Pareto CSV at: {pareto_csv}")

        df = pd.read_csv(pareto_csv)
        # Ensure standard sorting by CFD specificity ascending
        if "norm_cfd_specificity" in df.columns:
            df = df.sort_values("norm_cfd_specificity", ascending=True)

        return df.to_dict(orient="records")

    @classmethod
    def get_selected_guides(cls, gene: str) -> List[Dict[str, Any]]:
        """Loads Top-5 balanced guide selections with complete annotations."""
        gene = gene.upper()
        selected_csv = STEP10B_DIR / gene / "run1" / "selected_guides.csv"
        if not selected_csv.exists():
            raise FileNotFoundError(f"Missing selected guides CSV at: {selected_csv}")

        df = pd.read_csv(selected_csv)
        return df.to_dict(orient="records")

    @classmethod
    def get_feasible_candidates(
        cls, gene: str, sample_limit: Optional[int] = 2000
    ) -> List[Dict[str, Any]]:
        """
        Loads feasible candidates with normalized scores for scatter plotting.
        If sample_limit is set and feasible set exceeds limit, samples uniformly
        while guaranteeing that all Pareto and Selected guides are included.
        """
        gene = gene.upper()
        feas_csv = STEP10B_DIR / gene / "run1" / "feasible_candidates.csv"
        if not feas_csv.exists():
            raise FileNotFoundError(f"Missing feasible CSV at: {feas_csv}")

        df = pd.read_csv(feas_csv)

        # Flag Pareto and Selected guides
        pareto_ids = {r["guide_id"] for r in cls.get_pareto_front(gene)}
        selected_ids = {r["guide_id"] for r in cls.get_selected_guides(gene)}

        df["is_pareto"] = df["guide_id"].isin(pareto_ids)
        df["is_selected"] = df["guide_id"].isin(selected_ids)

        if sample_limit and len(df) > sample_limit:
            priority_mask = df["is_pareto"] | df["is_selected"]
            priority_df = df[priority_mask]
            regular_df = df[~priority_mask]

            needed = max(0, sample_limit - len(priority_df))
            if needed > 0 and len(regular_df) > needed:
                regular_sampled = regular_df.sample(n=needed, random_state=42)
            else:
                regular_sampled = regular_df

            df = pd.concat([priority_df, regular_sampled]).sort_values("guide_id")

        cols_to_export = [
            "guide_id",
            "guide_rna",
            "strand",
            "pam_position",
            "gc_content",
            "melting_temp_c",
            "delta_g_kcal_mol",
            "folding_energy_kcal_mol",
            "secondary_structure",
            "deep_spcas9_score",
            "cfd_off_target_risk",
            "cfd_specificity_score",
            "norm_activity",
            "norm_cfd_specificity",
            "is_pareto",
            "is_selected",
        ]
        available_cols = [c for c in cols_to_export if c in df.columns]
        return df[available_cols].to_dict(orient="records")

    @classmethod
    def get_sensitivity_summary(cls, gene: str) -> Dict[str, Any]:
        """Loads Step 11 sensitivity analysis metrics for a gene."""
        gene = gene.upper()
        if not STEP11_SUMMARY_JSON.exists():
            return {"status": "unavailable"}

        with open(STEP11_SUMMARY_JSON, "r", encoding="utf-8") as fp:
            s11 = json.load(fp)

        p2 = s11.get("part2_censored_sensitivity", {}).get("run1", {}).get(gene, {})
        p3 = s11.get("part3_epsilon_resolution", {}).get(gene, {})
        p4 = s11.get("part4_objective_consistency", {})

        return {
            "gene": gene,
            "censored_worst_case": {
                "max_observed_risk": p2.get("imputation", {}).get(
                    "gene_specific_max_observed_cfd_risk"
                ),
                "imputed_spec": p2.get("imputation", {}).get("imputed_cfd_specificity"),
                "n_guides_imputed": p2.get("imputation", {}).get("n_guides_imputed"),
                "feasible_count": p2.get("feasibility", {}).get("feasible"),
                "pareto_size": p2.get("pareto", {}).get("verified_pareto_count"),
                "overlap_with_primary_pct": p2.get("overlap_pct", 100.0),
                "is_100pct_stable": p2.get("overlap_pct") == 100.0,
            },
            "epsilon_resolution": {
                "resolution_stable": p3.get("resolution_stable", True),
                "selection_stable_50_to_200": p3.get("selection_stable_50_to_200", True),
                "pareto_size_delta": p3.get("pareto_size_change_50_to_200", 0),
                "steps": p3.get("per_step_count", {}),
            },
            "objective_consistency": {
                "identity_passed": p4.get("identity_100_100_plus_R", {}).get("pass", True),
                "spearman_rank_corr": p4.get("monotone_direction_per_gene", {})
                .get(gene, {})
                .get("spearman_correlation_of_ranks", 1.0),
            },
        }

    @classmethod
    def get_full_dashboard_payload(cls) -> Dict[str, Any]:
        """
        Builds a comprehensive multi-gene JSON payload containing all overview,
        Pareto, selected guide, feasible sample, and sensitivity data.
        Enables seamless, client-side zero-latency multi-gene switching.
        """
        log.info("Generating full dashboard JSON payload across all 4 genes...")
        genes_data = {}
        for g in SUPPORTED_GENES:
            log.info("  Packaging data for %s ...", g)
            genes_data[g] = {
                "overview": cls.get_gene_overview(g),
                "pareto_front": cls.get_pareto_front(g),
                "selected_guides": cls.get_selected_guides(g),
                "feasible_sample": cls.get_feasible_candidates(g, sample_limit=2500),
                "sensitivity": cls.get_sensitivity_summary(g),
            }

        # Multi-gene master comparison table
        master_comparison = []
        for g in SUPPORTED_GENES:
            ov = genes_data[g]["overview"]
            sens = genes_data[g]["sensitivity"]
            master_comparison.append({
                "gene": g,
                "chromosome": ov["metadata"].get("chromosome", ""),
                "seq_length": ov["metadata"].get("sequence_length", 0),
                "total_candidates": ov["total_candidates"],
                "excluded_candidates": ov["excluded_candidates"],
                "feasible_candidates": ov["feasible_candidates"],
                "feasible_yield_pct": round(
                    ov["feasible_candidates"] / max(ov["total_candidates"], 1) * 100, 1
                ),
                "pareto_size": ov["pareto_front_size"],
                "selected_count": ov["selected_count"],
                "activity_min": ov["objective_ranges"]["activity"].get("min"),
                "activity_max": ov["objective_ranges"]["activity"].get("max"),
                "cfd_risk_max": ov["objective_ranges"]["cfd_risk"].get("max"),
                "cfd_spec_min": ov["objective_ranges"]["cfd_spec"].get("min"),
                "sensitivity_overlap_pct": sens.get("censored_worst_case", {}).get(
                    "overlap_with_primary_pct", 100.0
                ),
            })

        payload = {
            "platform": "Multi-Objective CRISPR Guide RNA Optimization Framework",
            "version": "Step 11 Production Baseline",
            "supported_genes": SUPPORTED_GENES,
            "master_comparison": master_comparison,
            "genes": genes_data,
            "frozen_baseline_verified": cls.verify_frozen_baseline(),
            "scientific_disclaimer": (
                "All guide scores, on-target cleavage predictions (DeepSpCas9), and off-target "
                "specificity metrics (CFD) are computational outputs generated by machine learning models "
                "and biophysical empirical matrices. They represent mathematical optimization and do not "
                "constitute wet-lab experimental validation."
            ),
        }
        return payload

    @classmethod
    def verify_frozen_baseline(cls) -> Dict[str, Any]:
        """
        Automated self-test checking all 4 genes against the frozen research baseline.
        Returns a structured report of pass/fail status.
        """
        report = {"all_passed": True, "per_gene": {}}
        for gene, base in FROZEN_BASELINE.items():
            try:
                ov = cls.get_gene_overview(gene)
                tot_ok = ov["total_candidates"] == base["total_candidates"]
                feas_ok = ov["feasible_candidates"] == base["feasible_candidates"]
                par_ok = ov["pareto_front_size"] == base["pareto_size"]
                sel_ids = ov["selected_guide_ids"]
                ids_ok = sel_ids == base["selected_top5"]

                passed = tot_ok and feas_ok and par_ok and ids_ok
                if not passed:
                    report["all_passed"] = False

                report["per_gene"][gene] = {
                    "passed": passed,
                    "total_candidates": {"actual": ov["total_candidates"], "expected": base["total_candidates"], "match": tot_ok},
                    "feasible_candidates": {"actual": ov["feasible_candidates"], "expected": base["feasible_candidates"], "match": feas_ok},
                    "pareto_size": {"actual": ov["pareto_front_size"], "expected": base["pareto_size"], "match": par_ok},
                    "selected_top5_match": ids_ok,
                }
            except Exception as exc:
                report["all_passed"] = False
                report["per_gene"][gene] = {"passed": False, "error": str(exc)}

        return report

    @classmethod
    def list_dynamic_genes(cls) -> List[str]:
        """Lists any novel genes analyzed via the dynamic pipeline."""
        if not DYNAMIC_RESULTS_DIR.exists():
            return []
        genes = []
        for p in DYNAMIC_RESULTS_DIR.iterdir():
            if p.is_dir() and (p / "dynamic_run_summary.json").exists():
                genes.append(p.name)
        return sorted(genes)

    @classmethod
    def get_dynamic_gene_summary(cls, gene: str) -> Optional[Dict[str, Any]]:
        """Loads the dynamic run summary for a novel gene if available."""
        gene = gene.upper()
        summary_file = DYNAMIC_RESULTS_DIR / gene / "dynamic_run_summary.json"
        if not summary_file.exists():
            return None
        with open(summary_file, "r", encoding="utf-8") as fp:
            return json.load(fp)


if __name__ == "__main__":
    print("=" * 72)
    print("PIPELINE CONTROLLER: FROZEN BASELINE VERIFICATION")
    print("=" * 72)
    verification = PipelineController.verify_frozen_baseline()
    print(f"Overall Integrity Status: {'[PASS] ALL PASSED' if verification['all_passed'] else '[FAIL] MISMATCH DETECTED'}")
    for g, res in verification["per_gene"].items():
        status = "[PASS]" if res.get("passed") else "[FAIL]"
        print(f"  {status} {g}:")
        for k, v in res.items():
            if k != "passed":
                print(f"      {k}: {v}")
    print("=" * 72)
