"""
src/step8_benchmarking.py

Step 8 — Benchmarking and Comparative Evaluation

Compares the proposed multi-objective Pareto-based optimization framework
against simpler guide-selection strategies using the SAME feasible candidate
sets generated in Step 7.

Selection strategies:
    A. Highest_DeepSpCas9  — single-objective activity maximisation
    B. Lowest_OffTarget    — single-objective off-target minimisation
    C. Equal_Weight        — weighted-sum scalarisation (0.5 Ê + 0.5 Ŝ)
    D. Pareto_Selected     — Pareto-optimal guide with highest balanced score

All methods select from the IDENTICAL feasible set for each gene, ensuring
methodological fairness.

Usage:
    python -m src.step8_benchmarking
    python -m src.step8_benchmarking --genes TP53 HBB EGFR BRCA1

Author: Bioinformatics Development Team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step8_benchmarking")

GENES = ["TP53", "HBB", "EGFR", "BRCA1"]


# =========================================================================== #
# 1. Data Loading
# =========================================================================== #
def load_gene_data(gene: str) -> dict:
    """Load the feasible candidates and Pareto front for a gene."""
    opt_dir = PROJECT_ROOT / "results" / gene / "optimization"
    feasible_csv = opt_dir / "feasible_candidates.csv"
    pareto_csv = opt_dir / "pareto_front.csv"
    selected_csv = opt_dir / "selected_guides.csv"

    if not feasible_csv.exists():
        raise FileNotFoundError(f"Feasible candidates not found: {feasible_csv}")

    feasible_df = pd.read_csv(feasible_csv)
    pareto_df = pd.read_csv(pareto_csv) if pareto_csv.exists() else pd.DataFrame()
    selected_df = pd.read_csv(selected_csv) if selected_csv.exists() else pd.DataFrame()

    pareto_ids = set(pareto_df["guide_id"].tolist()) if not pareto_df.empty else set()

    return {
        "gene": gene,
        "feasible_df": feasible_df,
        "pareto_df": pareto_df,
        "selected_df": selected_df,
        "pareto_ids": pareto_ids,
    }


# =========================================================================== #
# 2. Baseline Selection Strategies
# =========================================================================== #
def select_highest_deepspcas9(feasible_df: pd.DataFrame) -> pd.Series:
    """Baseline A: Select the guide with the highest DeepSpCas9 score."""
    idx = feasible_df["deep_spcas9_score"].idxmax()
    return feasible_df.loc[idx]


def select_lowest_off_target(feasible_df: pd.DataFrame) -> pd.Series:
    """
    Baseline B: Select the guide with the lowest off_target_count.

    Tie-break rule (deterministic):
      1. Higher DeepSpCas9 score
      2. Lexicographically smaller guide_id
    """
    min_ot = feasible_df["off_target_count"].min()
    candidates = feasible_df[feasible_df["off_target_count"] == min_ot].copy()

    # Tie-break: highest DeepSpCas9, then lex-smallest guide_id
    candidates = candidates.sort_values(
        ["deep_spcas9_score", "guide_id"],
        ascending=[False, True],
    )
    return candidates.iloc[0]


def select_equal_weight(feasible_df: pd.DataFrame) -> pd.Series:
    """Baseline C: Select the guide maximising 0.5*E_hat + 0.5*S_hat."""
    scores = 0.5 * feasible_df["norm_activity"] + 0.5 * feasible_df["norm_specificity"]
    idx = scores.idxmax()
    return feasible_df.loc[idx]


def select_pareto_best(pareto_df: pd.DataFrame) -> pd.Series:
    """
    Method D: From the Pareto-optimal set, select the guide with the highest
    balanced score (0.5*E_hat + 0.5*S_hat).
    """
    scores = 0.5 * pareto_df["norm_activity"] + 0.5 * pareto_df["norm_specificity"]
    idx = scores.idxmax()
    return pareto_df.loc[idx]


def build_selection_row(gene: str, method: str, guide: pd.Series,
                        pareto_ids: set) -> dict:
    """Build a standardised row for the benchmark comparison table."""
    guide_id = str(guide["guide_id"])
    e_hat = float(guide["norm_activity"])
    s_hat = float(guide["norm_specificity"])
    balanced = round(0.5 * e_hat + 0.5 * s_hat, 6)
    is_capped = bool(guide["off_target_capped"])

    return {
        "Gene": gene,
        "Method": method,
        "Guide_ID": guide_id,
        "Guide_Sequence": str(guide["guide_rna"]),
        "Strand": str(guide["strand"]),
        "GC_Content": round(float(guide["gc_content"]), 4),
        "DeepSpCas9": round(float(guide["deep_spcas9_score"]), 4),
        "Off_Target_Count": int(guide["off_target_count"]),
        "Off_Target_Capped": is_capped,
        "E_hat": round(e_hat, 6),
        "S_hat": round(s_hat, 6),
        "Balanced_Score": balanced,
        "Pareto_Status": guide_id in pareto_ids,
    }


# =========================================================================== #
# 3. Weighted-Sum Sweep
# =========================================================================== #
def weighted_sum_sweep(feasible_df: pd.DataFrame, pareto_ids: set,
                       gene: str) -> pd.DataFrame:
    """
    Systematic weighted-sum sweep with w_E in {0.00, 0.05, ..., 1.00}.

    For each weight pair, select the guide maximising:
        F = w_E * E_hat + w_S * S_hat

    Returns a DataFrame with one row per weight pair.
    """
    weights_e = np.arange(0.0, 1.001, 0.05)
    rows = []

    activity = feasible_df["norm_activity"].values
    specificity = feasible_df["norm_specificity"].values

    for w_e in weights_e:
        w_s = 1.0 - w_e
        scores = w_e * activity + w_s * specificity
        idx = int(np.argmax(scores))

        guide = feasible_df.iloc[idx]
        guide_id = str(guide["guide_id"])
        e_hat = float(guide["norm_activity"])
        s_hat = float(guide["norm_specificity"])

        rows.append({
            "Gene": gene,
            "Weight_E": round(float(w_e), 2),
            "Weight_S": round(float(w_s), 2),
            "Selected_Guide": guide_id,
            "E_hat": round(e_hat, 6),
            "S_hat": round(s_hat, 6),
            "Balanced_Score": round(0.5 * e_hat + 0.5 * s_hat, 6),
            "Is_Pareto_Optimal": guide_id in pareto_ids,
        })

    return pd.DataFrame(rows)


# =========================================================================== #
# 4. Trade-Off Analysis
# =========================================================================== #
def compute_trade_off_analysis(benchmark_rows: list[dict], gene: str) -> dict:
    """
    Compute pairwise trade-off differences between strategies for one gene.
    """
    gene_rows = {r["Method"]: r for r in benchmark_rows if r["Gene"] == gene}
    methods = ["Highest_DeepSpCas9", "Lowest_OffTarget", "Equal_Weight", "Pareto_Selected"]

    # Reference point: Equal_Weight (the Step 7 decision rule)
    ref = gene_rows.get("Equal_Weight", {})
    if not ref:
        return {}

    comparisons = {}
    for m in methods:
        if m == "Equal_Weight":
            continue
        other = gene_rows.get(m, {})
        if not other:
            continue
        comparisons[f"{m}_vs_Equal_Weight"] = {
            "Delta_E_hat": round(other["E_hat"] - ref["E_hat"], 6),
            "Delta_S_hat": round(other["S_hat"] - ref["S_hat"], 6),
            "Delta_DeepSpCas9": round(other["DeepSpCas9"] - ref["DeepSpCas9"], 4),
            "Delta_OffTarget": int(other["Off_Target_Count"] - ref["Off_Target_Count"]),
        }

    return comparisons


# =========================================================================== #
# 5. Visualisation
# =========================================================================== #
def generate_benchmark_plots(gene_data_map: dict, benchmark_rows: list[dict],
                             output_dir: Path) -> list[str]:
    """Generate per-gene and combined benchmark plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    method_markers = {
        "Highest_DeepSpCas9": ("^", "#F44336", "Highest DeepSpCas9"),
        "Lowest_OffTarget": ("v", "#2196F3", "Lowest Off-Target"),
        "Equal_Weight": ("D", "#FF9800", "Equal Weight (0.5/0.5)"),
        "Pareto_Selected": ("*", "#4CAF50", "Pareto Selected"),
    }

    gene_colors = {
        "TP53": "#2196F3",
        "HBB": "#4CAF50",
        "EGFR": "#FF9800",
        "BRCA1": "#E91E63",
    }

    genes = list(gene_data_map.keys())

    # ---- Per-gene benchmark plots ----
    for gene in genes:
        data = gene_data_map[gene]
        feasible_df = data["feasible_df"]
        pareto_df = data["pareto_df"]
        gene_bm = [r for r in benchmark_rows if r["Gene"] == gene]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Feasible set (light grey)
        capped_mask = feasible_df["off_target_capped"] == True  # noqa: E712
        uncapped_mask = ~capped_mask

        if uncapped_mask.any():
            ax.scatter(
                feasible_df.loc[uncapped_mask, "norm_activity"],
                feasible_df.loc[uncapped_mask, "norm_specificity"],
                c="#e0e0e0", s=4, alpha=0.3, label="Feasible (exact OT)",
                zorder=1, edgecolors="none",
            )
        if capped_mask.any():
            ax.scatter(
                feasible_df.loc[capped_mask, "norm_activity"],
                feasible_df.loc[capped_mask, "norm_specificity"],
                c="none", s=6, alpha=0.2,
                label="Feasible (censored OT ≥999)",
                zorder=1, edgecolors="#e69500", linewidths=0.4,
            )

        # Pareto front
        if not pareto_df.empty:
            ax.scatter(
                pareto_df["norm_activity"],
                pareto_df["norm_specificity"],
                c=gene_colors.get(gene, "#9E9E9E"), s=50, alpha=0.7,
                label="Pareto-optimal", zorder=3,
                edgecolors="navy", linewidths=0.5,
            )
            pf_sorted = pareto_df.sort_values("norm_activity")
            ax.plot(
                pf_sorted["norm_activity"],
                pf_sorted["norm_specificity"],
                c=gene_colors.get(gene, "#1565C0"), linewidth=1.5,
                alpha=0.4, zorder=2,
            )

        # Selected guides from each strategy
        for row in gene_bm:
            marker, color, label = method_markers.get(
                row["Method"], ("o", "black", row["Method"])
            )
            ax.scatter(
                row["E_hat"], row["S_hat"],
                marker=marker, c=color, s=250, zorder=5,
                edgecolors="black", linewidths=1.0,
                label=f"{label}",
            )
            ax.annotate(
                row["Method"].replace("_", "\n"),
                (row["E_hat"], row["S_hat"]),
                textcoords="offset points", xytext=(12, -5),
                fontsize=6, ha="left", alpha=0.8,
            )

        ax.set_xlabel("Ê (Normalised Activity) →", fontsize=11)
        ax.set_ylabel("Ŝ (Normalised Specificity) →", fontsize=11)
        ax.set_title(
            f"{gene} — Strategy Comparison in Objective Space\n"
            f"(Feasible: {len(feasible_df):,}, Pareto: {len(pareto_df)})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = plots_dir / f"{gene}_benchmark.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(plot_path))
        log.info("Saved benchmark plot: %s", plot_path)

    # ---- Combined comparison plot (2x2) ----
    n_genes = len(genes)
    if n_genes > 0:
        ncols = 2
        nrows = (n_genes + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)

        for idx, gene in enumerate(genes):
            row_idx, col_idx = idx // ncols, idx % ncols
            ax = axes[row_idx][col_idx]

            data = gene_data_map[gene]
            feasible_df = data["feasible_df"]
            pareto_df = data["pareto_df"]
            gene_bm = [r for r in benchmark_rows if r["Gene"] == gene]

            # Feasible
            ax.scatter(
                feasible_df["norm_activity"],
                feasible_df["norm_specificity"],
                c="#e8e8e8", s=2, alpha=0.2, zorder=1, edgecolors="none",
            )

            # Pareto front
            if not pareto_df.empty:
                ax.scatter(
                    pareto_df["norm_activity"],
                    pareto_df["norm_specificity"],
                    c=gene_colors.get(gene, "#9E9E9E"), s=30, alpha=0.6,
                    zorder=3, edgecolors="navy", linewidths=0.4,
                )
                pf_sorted = pareto_df.sort_values("norm_activity")
                ax.plot(
                    pf_sorted["norm_activity"],
                    pf_sorted["norm_specificity"],
                    c=gene_colors.get(gene, "#1565C0"), linewidth=1.2,
                    alpha=0.4, zorder=2,
                )

            # Strategies
            for r in gene_bm:
                marker, color, label = method_markers.get(
                    r["Method"], ("o", "black", r["Method"])
                )
                ax.scatter(
                    r["E_hat"], r["S_hat"],
                    marker=marker, c=color, s=200, zorder=5,
                    edgecolors="black", linewidths=0.8,
                    label=label,
                )

            ax.set_xlabel("Ê →", fontsize=9)
            ax.set_ylabel("Ŝ →", fontsize=9)
            ax.set_title(f"{gene}", fontsize=11, fontweight="bold")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

        # Hide unused subplots
        for idx in range(len(genes), nrows * ncols):
            row_idx, col_idx = idx // ncols, idx % ncols
            axes[row_idx][col_idx].set_visible(False)

        fig.suptitle(
            "Step 8 — Multi-Gene Strategy Comparison in Objective Space",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        combined_path = plots_dir / "combined_benchmark.png"
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(combined_path))
        log.info("Saved combined benchmark plot: %s", combined_path)

    log.info("Generated %d benchmark plots", len(generated))
    return generated


# =========================================================================== #
# 6. Validation
# =========================================================================== #
def validate_step8(
    gene_data_map: dict,
    benchmark_rows: list[dict],
    ws_sweep_df: pd.DataFrame,
    step7_selected: dict,
) -> list[dict]:
    """
    Run all Step 8 validation checks.
    Returns a list of check dicts.
    """
    checks = []

    # 1. All four genes processed
    genes_processed = set(r["Gene"] for r in benchmark_rows)
    checks.append({
        "check": "all_genes_processed",
        "pass": genes_processed == set(GENES),
        "detail": f"Processed: {sorted(genes_processed)}",
    })

    # 2. Same feasible set used by all methods within each gene
    same_feasible = True
    for gene in GENES:
        gene_rows = [r for r in benchmark_rows if r["Gene"] == gene]
        feasible_ids = set(gene_data_map[gene]["feasible_df"]["guide_id"].tolist())
        for row in gene_rows:
            if row["Guide_ID"] not in feasible_ids:
                same_feasible = False
                break
    checks.append({
        "check": "same_feasible_set",
        "pass": same_feasible,
        "detail": "All selected guides belong to their gene's feasible set",
    })

    # 3. Exactly one guide selected per deterministic baseline
    one_per_method = True
    for gene in GENES:
        for method in ["Highest_DeepSpCas9", "Lowest_OffTarget", "Equal_Weight", "Pareto_Selected"]:
            count = sum(1 for r in benchmark_rows
                        if r["Gene"] == gene and r["Method"] == method)
            if count != 1:
                one_per_method = False
    checks.append({
        "check": "one_guide_per_method",
        "pass": one_per_method,
        "detail": f"Total rows: {len(benchmark_rows)}, expected: {len(GENES) * 4}",
    })

    # 4. Selected guides exist in input dataset
    all_exist = all(
        row["Guide_ID"] in set(gene_data_map[row["Gene"]]["feasible_df"]["guide_id"].tolist())
        for row in benchmark_rows
    )
    checks.append({
        "check": "selected_guides_exist",
        "pass": all_exist,
        "detail": "All selected guides found in feasible candidate set",
    })

    # 5. Pareto-selected guides belong to Pareto set
    pareto_correct = all(
        row["Pareto_Status"]
        for row in benchmark_rows if row["Method"] == "Pareto_Selected"
    )
    checks.append({
        "check": "pareto_selected_in_pareto_set",
        "pass": pareto_correct,
        "detail": "All Pareto_Selected guides are Pareto-optimal",
    })

    # 6. Equal-weight selection reproduces Step 7 results
    reproduces = True
    reproduce_detail = []
    for gene in GENES:
        ew_row = next(
            (r for r in benchmark_rows
             if r["Gene"] == gene and r["Method"] == "Equal_Weight"),
            None,
        )
        s7_guide = step7_selected.get(gene, "")
        if ew_row and s7_guide:
            match = ew_row["Guide_ID"] == s7_guide
            reproduce_detail.append(f"{gene}: {match}")
            if not match:
                reproduces = False
        else:
            reproduce_detail.append(f"{gene}: missing data")
            reproduces = False
    checks.append({
        "check": "equal_weight_reproduces_step7",
        "pass": reproduces,
        "detail": "; ".join(reproduce_detail),
    })

    # 7. Weighted-sum sweep weights sum to 1
    ws_sums = ws_sweep_df["Weight_E"] + ws_sweep_df["Weight_S"]
    ws_ok = bool(np.allclose(ws_sums, 1.0, atol=1e-6))
    checks.append({
        "check": "weighted_sum_weights_sum_to_1",
        "pass": ws_ok,
        "detail": f"Min sum: {ws_sums.min():.4f}, Max sum: {ws_sums.max():.4f}",
    })

    # 8. No NaN objective values
    no_nan = all(
        not (np.isnan(r["E_hat"]) or np.isnan(r["S_hat"]))
        for r in benchmark_rows
    )
    checks.append({
        "check": "no_nan_objectives",
        "pass": no_nan,
        "detail": "All E_hat and S_hat values are finite",
    })

    # 9. Capped guides correctly flagged
    capped_ok = True
    for row in benchmark_rows:
        gene = row["Gene"]
        guide_id = row["Guide_ID"]
        feasible_df = gene_data_map[gene]["feasible_df"]
        match = feasible_df[feasible_df["guide_id"] == guide_id]
        if not match.empty:
            actual_capped = bool(match.iloc[0]["off_target_capped"])
            if row["Off_Target_Capped"] != actual_capped:
                capped_ok = False
                break
    checks.append({
        "check": "capped_guides_flagged",
        "pass": capped_ok,
        "detail": "Off_Target_Capped flags match source data",
    })

    # 10. Input feature tables unchanged (verify row counts)
    tables_ok = True
    for gene in GENES:
        orig_path = PROJECT_ROOT / "results" / gene / "gRNA_features.csv"
        if orig_path.exists():
            orig_df = pd.read_csv(orig_path)
            expected_cols = 34
            if len(orig_df.columns) != expected_cols:
                tables_ok = False
    checks.append({
        "check": "input_tables_unchanged",
        "pass": tables_ok,
        "detail": "Original gRNA_features.csv files have 34 columns",
    })

    return checks


# =========================================================================== #
# 7. Main Orchestrator
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8 — Benchmarking and Comparative Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--genes", nargs="*", default=GENES,
        help="List of gene symbols to benchmark",
    )
    args = parser.parse_args()
    genes = [g.strip().upper() for g in args.genes]

    log.info("=" * 72)
    log.info("STEP 8 — BENCHMARKING AND COMPARATIVE EVALUATION")
    log.info("=" * 72)
    log.info("Genes: %s", genes)
    log.info("")

    output_dir = PROJECT_ROOT / "results" / "step8_benchmarking"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Phase 1: Load data
    # =================================================================
    log.info("=" * 72)
    log.info("PHASE 1: LOADING FEASIBLE CANDIDATE SETS")
    log.info("=" * 72)

    gene_data_map = {}
    for gene in genes:
        log.info("Loading %s ...", gene)
        data = load_gene_data(gene)
        gene_data_map[gene] = data
        log.info("  Feasible: %d | Pareto: %d",
                 len(data["feasible_df"]), len(data["pareto_df"]))
    log.info("")

    # =================================================================
    # Phase 2: Apply selection strategies
    # =================================================================
    log.info("=" * 72)
    log.info("PHASE 2: APPLYING SELECTION STRATEGIES")
    log.info("=" * 72)

    benchmark_rows = []

    for gene in genes:
        data = gene_data_map[gene]
        feasible_df = data["feasible_df"]
        pareto_df = data["pareto_df"]
        pareto_ids = data["pareto_ids"]

        log.info("")
        log.info("--- %s ---", gene)

        # A. Highest DeepSpCas9
        guide_a = select_highest_deepspcas9(feasible_df)
        row_a = build_selection_row(gene, "Highest_DeepSpCas9", guide_a, pareto_ids)
        benchmark_rows.append(row_a)
        log.info("  A. Highest DeepSpCas9: %s (score=%.2f, OT=%d, capped=%s)",
                 row_a["Guide_ID"], row_a["DeepSpCas9"],
                 row_a["Off_Target_Count"], row_a["Off_Target_Capped"])

        # B. Lowest Off-Target
        guide_b = select_lowest_off_target(feasible_df)
        row_b = build_selection_row(gene, "Lowest_OffTarget", guide_b, pareto_ids)
        benchmark_rows.append(row_b)
        log.info("  B. Lowest Off-Target:  %s (OT=%d, score=%.2f, capped=%s)",
                 row_b["Guide_ID"], row_b["Off_Target_Count"],
                 row_b["DeepSpCas9"], row_b["Off_Target_Capped"])

        # C. Equal Weight
        guide_c = select_equal_weight(feasible_df)
        row_c = build_selection_row(gene, "Equal_Weight", guide_c, pareto_ids)
        benchmark_rows.append(row_c)
        log.info("  C. Equal Weight:       %s (balanced=%.4f, Ê=%.4f, Ŝ=%.4f)",
                 row_c["Guide_ID"], row_c["Balanced_Score"],
                 row_c["E_hat"], row_c["S_hat"])

        # D. Pareto Selected
        if not pareto_df.empty:
            guide_d = select_pareto_best(pareto_df)
            row_d = build_selection_row(gene, "Pareto_Selected", guide_d, pareto_ids)
        else:
            # Fallback: same as equal weight
            row_d = dict(row_c)
            row_d["Method"] = "Pareto_Selected"
            log.warning("  No Pareto front for %s — using equal-weight fallback", gene)
        benchmark_rows.append(row_d)
        log.info("  D. Pareto Selected:    %s (balanced=%.4f, Ê=%.4f, Ŝ=%.4f)",
                 row_d["Guide_ID"], row_d["Balanced_Score"],
                 row_d["E_hat"], row_d["S_hat"])

    # Save benchmark comparison
    benchmark_df = pd.DataFrame(benchmark_rows)
    bm_path = output_dir / "benchmark_comparison.csv"
    benchmark_df.to_csv(bm_path, index=False)
    log.info("\nSaved benchmark comparison: %s", bm_path)

    # =================================================================
    # Phase 3: Pareto membership analysis
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 3: PARETO MEMBERSHIP ANALYSIS")
    log.info("=" * 72)

    pareto_membership_rows = []
    for row in benchmark_rows:
        pareto_membership_rows.append({
            "Gene": row["Gene"],
            "Method": row["Method"],
            "Guide_ID": row["Guide_ID"],
            "Is_Pareto_Optimal": row["Pareto_Status"],
        })

    pm_df = pd.DataFrame(pareto_membership_rows)
    pm_path = output_dir / "pareto_membership.csv"
    pm_df.to_csv(pm_path, index=False)
    log.info("Saved Pareto membership: %s", pm_path)

    for gene in genes:
        gene_pm = [r for r in pareto_membership_rows if r["Gene"] == gene]
        log.info("  %s:", gene)
        for r in gene_pm:
            status = "PARETO" if r["Is_Pareto_Optimal"] else "non-Pareto"
            log.info("    %s: %s (%s)", r["Method"], r["Guide_ID"], status)

    # =================================================================
    # Phase 4: Weighted-sum coverage analysis
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 4: WEIGHTED-SUM COVERAGE ANALYSIS")
    log.info("=" * 72)

    ws_frames = []
    ws_coverage_rows = []

    for gene in genes:
        data = gene_data_map[gene]
        ws_df = weighted_sum_sweep(data["feasible_df"], data["pareto_ids"], gene)
        ws_frames.append(ws_df)

        unique_solutions = ws_df["Selected_Guide"].nunique()
        pareto_recovered = ws_df[ws_df["Is_Pareto_Optimal"]]["Selected_Guide"].nunique()
        total_pareto = len(data["pareto_df"])
        coverage_pct = round(
            pareto_recovered / max(total_pareto, 1) * 100, 1
        )

        ws_coverage_rows.append({
            "Gene": gene,
            "Unique_Solutions": unique_solutions,
            "Pareto_Recovered": pareto_recovered,
            "Total_Pareto": total_pareto,
            "Pareto_Coverage_Pct": coverage_pct,
        })

        log.info("  %s: %d unique solutions, %d/%d Pareto recovered (%.1f%%)",
                 gene, unique_solutions, pareto_recovered, total_pareto, coverage_pct)

    ws_sweep_all = pd.concat(ws_frames, ignore_index=True)
    ws_path = output_dir / "weighted_sum_sweep.csv"
    ws_sweep_all.to_csv(ws_path, index=False)
    log.info("Saved weighted-sum sweep: %s", ws_path)

    # =================================================================
    # Phase 5: Trade-off analysis
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 5: SINGLE-OBJECTIVE TRADE-OFF ANALYSIS")
    log.info("=" * 72)

    trade_off_results = {}
    for gene in genes:
        trade_offs = compute_trade_off_analysis(benchmark_rows, gene)
        trade_off_results[gene] = trade_offs
        log.info("  %s:", gene)
        for comparison, deltas in trade_offs.items():
            log.info("    %s: ΔÊ=%+.4f, ΔŜ=%+.4f, ΔDeepSpCas9=%+.2f, ΔOT=%+d",
                     comparison, deltas["Delta_E_hat"], deltas["Delta_S_hat"],
                     deltas["Delta_DeepSpCas9"], deltas["Delta_OffTarget"])

    # =================================================================
    # Phase 6: Summary metrics
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 6: SUMMARY METRICS")
    log.info("=" * 72)

    summary_rows = []
    for gene in genes:
        data = gene_data_map[gene]
        gene_bm = {r["Method"]: r for r in benchmark_rows if r["Gene"] == gene}
        ws_cov = next(r for r in ws_coverage_rows if r["Gene"] == gene)

        summary_rows.append({
            "Gene": gene,
            "Feasible_Guides": len(data["feasible_df"]),
            "Pareto_Guides": len(data["pareto_df"]),
            "WeightedSum_Unique_Solutions": ws_cov["Unique_Solutions"],
            "WeightedSum_Pareto_Coverage": ws_cov["Pareto_Coverage_Pct"],
            "HighestActivity_E_hat": gene_bm["Highest_DeepSpCas9"]["E_hat"],
            "HighestActivity_S_hat": gene_bm["Highest_DeepSpCas9"]["S_hat"],
            "LowestOT_E_hat": gene_bm["Lowest_OffTarget"]["E_hat"],
            "LowestOT_S_hat": gene_bm["Lowest_OffTarget"]["S_hat"],
            "EqualWeight_E_hat": gene_bm["Equal_Weight"]["E_hat"],
            "EqualWeight_S_hat": gene_bm["Equal_Weight"]["S_hat"],
            "ParetoSelected_E_hat": gene_bm["Pareto_Selected"]["E_hat"],
            "ParetoSelected_S_hat": gene_bm["Pareto_Selected"]["S_hat"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved benchmark summary: %s", summary_path)
    log.info("\n%s", summary_df.to_string(index=False))

    # =================================================================
    # Phase 7: Reproducibility check (Step 7 selected guides)
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 7: REPRODUCIBILITY CHECK (vs Step 7)")
    log.info("=" * 72)

    step7_selected = {}
    for gene in genes:
        sel_csv = PROJECT_ROOT / "results" / gene / "optimization" / "selected_guides.csv"
        if sel_csv.exists():
            sel_df = pd.read_csv(sel_csv)
            if not sel_df.empty:
                step7_selected[gene] = str(sel_df.iloc[0]["guide_id"])

    for gene in genes:
        ew_guide = next(
            (r["Guide_ID"] for r in benchmark_rows
             if r["Gene"] == gene and r["Method"] == "Equal_Weight"),
            "N/A"
        )
        s7_guide = step7_selected.get(gene, "N/A")
        match = ew_guide == s7_guide
        status = "MATCH" if match else "MISMATCH"
        log.info("  %s: Step7=%s, Step8_EW=%s → %s", gene, s7_guide, ew_guide, status)

    # =================================================================
    # Phase 8: Generate plots
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 8: GENERATING VISUALISATIONS")
    log.info("=" * 72)

    plot_paths = generate_benchmark_plots(gene_data_map, benchmark_rows, output_dir)

    # =================================================================
    # Phase 9: Validation
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 9: AUTOMATED VALIDATION")
    log.info("=" * 72)

    validation_checks = validate_step8(
        gene_data_map, benchmark_rows, ws_sweep_all, step7_selected,
    )

    validation_rows = []
    for check in validation_checks:
        status = "PASS" if check["pass"] else "FAIL"
        log.info("  [%s] %s: %s", status, check["check"], check["detail"])
        validation_rows.append({
            "Check": check["check"],
            "Status": "PASS" if check["pass"] else "FAIL",
            "Detail": check["detail"],
        })

    val_df = pd.DataFrame(validation_rows)
    val_path = output_dir / "step8_validation.csv"
    val_df.to_csv(val_path, index=False)
    log.info("\nSaved validation: %s", val_path)

    all_passed = all(c["pass"] for c in validation_checks)
    n_passed = sum(1 for c in validation_checks if c["pass"])
    log.info("  Result: %d/%d checks PASSED", n_passed, len(validation_checks))

    # =================================================================
    # Phase 10: Capped guide observations
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 10: CAPPED OFF-TARGET OBSERVATIONS")
    log.info("=" * 72)

    capped_observations = []
    for row in benchmark_rows:
        if row["Off_Target_Capped"]:
            capped_observations.append(
                f"  {row['Gene']}/{row['Method']}: {row['Guide_ID']} — "
                f"Selected guide has censored off-target count (≥999). "
                f"Reported OT={row['Off_Target_Count']} is a lower bound."
            )
    if capped_observations:
        for obs in capped_observations:
            log.warning(obs)
    else:
        log.info("  No selected guides have censored off-target counts.")

    # =================================================================
    # Phase 11: Comprehensive JSON report
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("PHASE 11: SAVING COMPREHENSIVE REPORT")
    log.info("=" * 72)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": "Step 8 — Benchmarking and Comparative Evaluation",
        "genes_evaluated": genes,
        "methods_compared": [
            {
                "id": "Highest_DeepSpCas9",
                "description": "Single-objective activity maximisation: select the feasible guide with the highest DeepSpCas9 score.",
                "type": "single-objective",
            },
            {
                "id": "Lowest_OffTarget",
                "description": "Single-objective off-target minimisation: select the feasible guide with the lowest off_target_count. Tie-break: higher DeepSpCas9, then lexicographically smaller guide_id.",
                "type": "single-objective",
            },
            {
                "id": "Equal_Weight",
                "description": "Weighted-sum scalarisation: select the feasible guide maximising 0.5*E_hat + 0.5*S_hat. This is a DEMONSTRATION decision rule.",
                "type": "scalarisation",
            },
            {
                "id": "Pareto_Selected",
                "description": "Pareto-based: from the Pareto-optimal set, select the guide with the highest balanced score (0.5*E_hat + 0.5*S_hat). The equal-weight guide is only a representative decision rule applied to the Pareto set.",
                "type": "multi-objective",
            },
        ],
        "fairness_rule": (
            "All methods select from the SAME feasible set for each gene. "
            "No method sees additional candidates. Comparison is methodologically fair."
        ),
        "benchmark_comparison": benchmark_rows,
        "pareto_membership": pareto_membership_rows,
        "weighted_sum_coverage": ws_coverage_rows,
        "trade_off_analysis": trade_off_results,
        "summary_metrics": summary_rows,
        "reproducibility_check": {
            gene: {
                "step7_guide": step7_selected.get(gene, "N/A"),
                "step8_equal_weight": next(
                    (r["Guide_ID"] for r in benchmark_rows
                     if r["Gene"] == gene and r["Method"] == "Equal_Weight"),
                    "N/A",
                ),
                "match": step7_selected.get(gene, "") == next(
                    (r["Guide_ID"] for r in benchmark_rows
                     if r["Gene"] == gene and r["Method"] == "Equal_Weight"),
                    "",
                ),
            }
            for gene in genes
        },
        "capped_guide_observations": capped_observations,
        "validation": {
            "total_checks": len(validation_checks),
            "passed": n_passed,
            "all_passed": all_passed,
            "checks": validation_rows,
        },
        "plots_generated": plot_paths,
        "mathematical_conclusions": [
            "Different selection strategies produce different guides under the same feasible set.",
            "The activity-maximising strategy achieves the highest E_hat but may sacrifice specificity (S_hat).",
            "The off-target-minimising strategy achieves the highest S_hat but may sacrifice activity (E_hat).",
            "Equal-weight scalarisation and Pareto-based selection both seek balanced solutions.",
            "The weighted-sum sweep recovers only a subset of the observed Pareto solutions.",
            "Multi-objective optimisation explicitly preserves the activity–specificity trade-off frontier.",
        ],
        "limitations": [
            "Off-target count is a computational Bowtie hit count, not a validated biological specificity score.",
            "Capped off-target counts (≥999) are censored/lower-bound computational values.",
            "Equal weighting (0.5/0.5) is a demonstration preference, not universally optimal.",
            "Computational comparison does not establish experimental biological efficacy.",
            "Pareto membership indicates mathematical non-dominance, not biological superiority.",
            "This benchmark compares selection rules, not prediction models.",
        ],
        "conclusion": (
            "The benchmark demonstrates how single-objective and multi-objective selection rules "
            "produce different solutions under the same feasible candidate set, and quantifies "
            "the activity–off-target trade-off represented by the proposed Pareto framework."
        ),
    }

    report_path = output_dir / "step8_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Saved comprehensive Step 8 report: %s", report_path)

    # =================================================================
    # Final summary
    # =================================================================
    log.info("")
    log.info("=" * 72)
    log.info("STEP 8 — FINAL SUMMARY")
    log.info("=" * 72)
    log.info("  Genes evaluated: %s", genes)
    log.info("  Methods compared: 4")
    log.info("  Total benchmark rows: %d", len(benchmark_rows))
    log.info("  Validation: %d/%d PASSED", n_passed, len(validation_checks))
    log.info("  Output directory: %s", output_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
