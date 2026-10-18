"""
src/convex_visualization.py

Visualization & Analysis Module — Convex LP Optimization Results
CRISPR Guide RNA Design Framework
==================================================================

Reads the three pipeline artefacts and produces six publication-quality
figures plus a summary CSV.

Inputs
------
  data/optimized_guides.csv              Module 4 weighted-sum ranking (2860 guides)
  data/convex_optimized_guides.csv       Module 5 LP-selected guides   (5 guides)
  data/gRNA_features_specificity.csv     Raw feature table              (2860 guides)

Outputs  (all written to results/convex_figures/)
-------
  fig1_objective_coefficients.png        Bar chart — c_i for LP-selected guides
  fig2_constraint_breakdown.png          Stacked bar — pass/fail per constraint
  fig3_efficiency_vs_specificity.png     Scatter — all 2860 + 5 LP guides highlighted
  fig4_weighted_vs_convex.png            Rank comparison between Module 4 and Module 5
  fig5_objective_breakdown.png           Grouped bar — E / S / R scores per LP guide
  fig6_constraint_funnel.png             Funnel waterfall — guides surviving each filter

  results/convex_optimization_summary.csv

Do NOT modify any input CSV or existing module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # non-interactive backend (safe on Windows / headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convex_visualization")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results") / "convex_figures"

IN_WEIGHTED  = DATA_DIR / "optimized_guides.csv"
IN_CONVEX    = DATA_DIR / "convex_optimized_guides.csv"
IN_FEATURES  = DATA_DIR / "gRNA_features_specificity.csv"

OUT_SUMMARY  = Path("results") / "convex_optimization_summary.csv"

# ---------------------------------------------------------------------------
# Biological constraint bounds (must match convex_optimization.py)
# ---------------------------------------------------------------------------
GC_LOW,  GC_HIGH  = 0.40, 0.60
TM_LOW,  TM_HIGH  = 55.0, 65.0
OT_MAX            = 20

# ---------------------------------------------------------------------------
# Aesthetic constants
# ---------------------------------------------------------------------------
PALETTE = {
    "convex":    "#2E86AB",   # steel blue  — LP-selected guides
    "weighted":  "#E84855",   # crimson     — weighted-rank guides
    "all":       "#A8DADC",   # pale cyan   — background population
    "pass":      "#57CC99",   # mint green  — constraint pass
    "fail":      "#EF233C",   # vivid red   — constraint fail
    "eff":       "#4CC9F0",   # sky blue    — efficiency
    "spec":      "#7209B7",   # purple      — specificity
    "struct":    "#F72585",   # magenta     — structural
}
FONT_TITLE  = {"fontsize": 14, "fontweight": "bold"}
FONT_LABEL  = {"fontsize": 11}
FONT_TICK   = {"labelsize": 9}
DPI         = 150


def _save(fig: plt.Figure, fname: str) -> None:
    path = RESULTS_DIR / fname
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ===========================================================================
# Figure 1 — Objective Coefficients of LP-selected guides
# ===========================================================================
def fig1_objective_coefficients(convex: pd.DataFrame) -> None:
    """Horizontal bar chart of c_i for each selected guide, sorted ascending."""
    df = convex.sort_values("objective_coefficient")
    labels = [f"Rank {int(r)}\n{g[:12]}..." for r, g in
              zip(df["rank"], df["guide_rna"])]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(
        labels, df["objective_coefficient"],
        color=PALETTE["convex"], edgecolor="white", height=0.55
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color="#333333")
    ax.set_xlabel("Objective coefficient  c_i = 0.35·E + 0.40·S + 0.25·R",
                  **FONT_LABEL)
    ax.set_title("Fig 1 — Objective Coefficients of LP-Selected Guides (K=10)",
                 **FONT_TITLE)
    ax.set_xlim(0, 1.08)
    ax.axvline(df["objective_coefficient"].mean(), color="#E84855",
               linestyle="--", linewidth=1.2, label=f"Mean c_i = {df['objective_coefficient'].mean():.4f}")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    _save(fig, "fig1_objective_coefficients.png")


# ===========================================================================
# Figure 2 — Feasible vs Rejected guides per constraint
# ===========================================================================
def fig2_constraint_breakdown(features: pd.DataFrame) -> None:
    """Stacked horizontal bar showing pass/fail count per biological constraint."""
    n = len(features)

    gc_pass  = features["gc_content"].between(GC_LOW, GC_HIGH).sum()
    tm_pass  = features["melting_temp_c"].between(TM_LOW, TM_HIGH).sum()
    ot_pass  = (features["off_target_count"] <= OT_MAX).sum()
    cap_pass = (~features["off_target_capped"].astype(bool)).sum()

    gc_pass_not_capped  = (features["gc_content"].between(GC_LOW, GC_HIGH) &
                           (features["off_target_count"] <= OT_MAX) &
                           ~features["off_target_capped"].astype(bool)).sum()
    all_pass = (
        features["gc_content"].between(GC_LOW, GC_HIGH) &
        features["melting_temp_c"].between(TM_LOW, TM_HIGH) &
        (features["off_target_count"] <= OT_MAX) &
        ~features["off_target_capped"].astype(bool)
    ).sum()

    constraints = [
        f"GC in [{GC_LOW:.2f}, {GC_HIGH:.2f}]",
        f"Tm in [{TM_LOW:.0f}, {TM_HIGH:.0f}] C",
        f"off-target count <= {OT_MAX}",
        "off_target_capped = False",
        "ALL constraints",
    ]
    passes = [gc_pass, tm_pass, ot_pass, cap_pass, all_pass]
    fails  = [n - p for p in passes]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(constraints))
    ax.barh(y, passes, color=PALETTE["pass"], label="Pass", height=0.5)
    ax.barh(y, fails,  left=passes, color=PALETTE["fail"], label="Fail", height=0.5)

    for i, (p, f) in enumerate(zip(passes, fails)):
        ax.text(p / 2, i, f"{p}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(p + f / 2, i, f"{f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(constraints, fontsize=9)
    ax.set_xlabel("Number of guides", **FONT_LABEL)
    ax.set_title("Fig 2 — Feasibility Breakdown per Biological Constraint  (n=2860)",
                 **FONT_TITLE)
    ax.axvline(n, color="#555", linestyle=":", linewidth=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, n * 1.05)
    fig.tight_layout()
    _save(fig, "fig2_constraint_breakdown.png")


# ===========================================================================
# Figure 3 — Efficiency vs Specificity scatter (all + LP highlights)
# ===========================================================================
def fig3_efficiency_vs_specificity(
    features: pd.DataFrame,
    convex: pd.DataFrame,
    weighted: pd.DataFrame,
) -> None:
    """
    Scatter plot of all 2860 guides (E vs S scores computed inline),
    with LP-selected and top-5 weighted guides highlighted.
    Uses the same min-max normalisation as the pipeline modules.
    """
    # Compute normalised Doench score for all guides (proxy for efficiency_score)
    lo, hi = features["doench2016_proxy_score"].min(), features["doench2016_proxy_score"].max()
    eff_all = (features["doench2016_proxy_score"] - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=features.index)

    # off_target_count_norm (inverted; lower count → higher score)
    lo_ot, hi_ot = features["off_target_count"].min(), features["off_target_count"].max()
    spec_proxy = 1.0 - (features["off_target_count"] - lo_ot) / (hi_ot - lo_ot) if hi_ot > lo_ot else pd.Series(0.5, index=features.index)

    # Feasible flag
    feasible = (
        features["gc_content"].between(GC_LOW, GC_HIGH) &
        features["melting_temp_c"].between(TM_LOW, TM_HIGH) &
        (features["off_target_count"] <= OT_MAX) &
        ~features["off_target_capped"].astype(bool)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # Background population
    ax.scatter(eff_all[~feasible], spec_proxy[~feasible],
               s=8, alpha=0.25, color=PALETTE["all"], label="Infeasible guides",
               linewidths=0, zorder=1)
    ax.scatter(eff_all[feasible], spec_proxy[feasible],
               s=40, alpha=0.7, color=PALETTE["pass"], edgecolors="#2d6a4f",
               linewidths=0.5, label="Feasible (biological constraints)", zorder=2)

    # Top-5 weighted guides
    top5_weighted = weighted.head(5)
    feat_weighted = features[features["guide_rna"].isin(top5_weighted["guide_rna"])].copy()
    eff_w  = (feat_weighted["doench2016_proxy_score"] - lo) / (hi - lo)
    spec_w = 1.0 - (feat_weighted["off_target_count"] - lo_ot) / (hi_ot - lo_ot)
    ax.scatter(eff_w, spec_w, s=110, color=PALETTE["weighted"],
               edgecolors="white", linewidths=1.2, zorder=4,
               label="Top-5 weighted-rank guides", marker="D")

    # LP-selected guides
    feat_conv = features[features["guide_rna"].isin(convex["guide_rna"])].copy()
    eff_c  = (feat_conv["doench2016_proxy_score"] - lo) / (hi - lo)
    spec_c = 1.0 - (feat_conv["off_target_count"] - lo_ot) / (hi_ot - lo_ot)
    ax.scatter(eff_c, spec_c, s=160, color=PALETTE["convex"],
               edgecolors="white", linewidths=1.5, zorder=5,
               label="LP-selected guides (convex opt.)", marker="*")

    # Annotate LP guides
    for idx, row in feat_conv.iterrows():
        e = float((row["doench2016_proxy_score"] - lo) / (hi - lo))
        s = float(1.0 - (row["off_target_count"] - lo_ot) / (hi_ot - lo_ot))
        ax.annotate(
            row["guide_rna"][:10] + "..",
            (e, s), textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color="#1a3a5c",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.7),
        )

    ax.set_xlabel("Efficiency score (normalised Doench proxy)", **FONT_LABEL)
    ax.set_ylabel("Specificity proxy (1 - normalised off-target count)", **FONT_LABEL)
    ax.set_title("Fig 3 — Efficiency vs Specificity for All 2860 Candidate Guides",
                 **FONT_TITLE)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(-0.02, 1.10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    _save(fig, "fig3_efficiency_vs_specificity.png")


# ===========================================================================
# Figure 4 — Weighted ranking vs Convex selection comparison
# ===========================================================================
def fig4_weighted_vs_convex(
    weighted: pd.DataFrame,
    convex: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """
    Dual-panel comparison:
      Left  — top-10 weighted-rank guides by final_score
      Right — 5 LP-selected guides by objective_coefficient
    Both panels include guide labels and score bars.
    """
    top10 = weighted.head(10).copy()
    convex_sorted = convex.sort_values("rank").copy()

    # Short labels
    top10["label"]         = [f"W-{r}: {g[:10]}.." for r, g in zip(top10["rank"], top10["guide_rna"])]
    convex_sorted["label"] = [f"LP-{r}: {g[:10]}.." for r, g in zip(convex_sorted["rank"], convex_sorted["guide_rna"])]

    # Mark overlap
    convex_guides = set(convex["guide_rna"])
    weighted_guides_top10 = set(top10["guide_rna"])
    overlap = convex_guides & weighted_guides_top10

    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(14, 6),
                                      gridspec_kw={"wspace": 0.55})

    # Left — weighted ranking
    colors_w = [PALETTE["convex"] if g in convex_guides else PALETTE["weighted"]
                for g in top10["guide_rna"]]
    bars_w = ax_w.barh(top10["label"][::-1], top10["final_score"][::-1],
                       color=colors_w[::-1], edgecolor="white", height=0.6)
    ax_w.bar_label(bars_w, fmt="%.4f", padding=3, fontsize=8)
    ax_w.set_xlim(0, 1.12)
    ax_w.set_xlabel("Weighted final score", **FONT_LABEL)
    ax_w.set_title("Module 4\nTop-10 Weighted-Rank Guides", **FONT_TITLE)
    ax_w.spines[["top", "right"]].set_visible(False)
    ax_w.tick_params(labelsize=9)

    # Right — convex LP
    colors_c = [PALETTE["weighted"] if g in weighted_guides_top10 else PALETTE["convex"]
                for g in convex_sorted["guide_rna"]]
    bars_c = ax_c.barh(convex_sorted["label"][::-1],
                       convex_sorted["objective_coefficient"][::-1],
                       color=colors_c[::-1], edgecolor="white", height=0.6)
    ax_c.bar_label(bars_c, fmt="%.4f", padding=3, fontsize=8)
    ax_c.set_xlim(0, 1.12)
    ax_c.set_xlabel("Objective coefficient  c_i", **FONT_LABEL)
    ax_c.set_title("Module 5\nLP Convex-Selected Guides", **FONT_TITLE)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.tick_params(labelsize=9)

    # Legend
    patch_both   = mpatches.Patch(color=PALETTE["convex"],   label="LP-selected guide")
    patch_w_only = mpatches.Patch(color=PALETTE["weighted"], label="Weighted-only / overlap")
    fig.legend(handles=[patch_both, patch_w_only], loc="lower center",
               ncol=2, fontsize=9, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.03))

    overlap_note = f"Overlap (appear in both): {len(overlap)} guide(s)"
    fig.text(0.5, 1.01, overlap_note, ha="center", fontsize=10,
             color="#444", style="italic")
    fig.suptitle("Fig 4 — Weighted Ranking vs Convex LP Selection Comparison",
                 **FONT_TITLE, y=1.07)
    fig.tight_layout()
    _save(fig, "fig4_weighted_vs_convex.png")


# ===========================================================================
# Figure 5 — Objective breakdown (E / S / R) per LP guide
# ===========================================================================
def fig5_objective_breakdown(convex: pd.DataFrame) -> None:
    """Grouped bar chart: efficiency, specificity, structural per LP guide."""
    df = convex.sort_values("rank").reset_index(drop=True)
    labels = [f"Rank {int(r)}\n{g[:12]}.." for r, g in zip(df["rank"], df["guide_rna"])]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width, df["efficiency_score"],
                width, label="Efficiency (E)",  color=PALETTE["eff"],   edgecolor="white")
    b2 = ax.bar(x,         df["specificity_score"],
                width, label="Specificity (S)", color=PALETTE["spec"],  edgecolor="white")
    b3 = ax.bar(x + width, df["structural_suitability_score"],
                width, label="Structural (R)",  color=PALETTE["struct"], edgecolor="white")

    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, rotation=90)

    # Objective coefficient as a line overlay
    ax.plot(x, df["objective_coefficient"], marker="o", color="#333",
            linewidth=1.5, markersize=6, label="c_i  (0.35E + 0.40S + 0.25R)",
            zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Normalised score  [0, 1]", **FONT_LABEL)
    ax.set_title(
        "Fig 5 — Objective Score Breakdown for LP-Selected Guides\n"
        "Weights: Efficiency 0.35 | Specificity 0.40 | Structural 0.25",
        **FONT_TITLE
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    _save(fig, "fig5_objective_breakdown.png")


# ===========================================================================
# Figure 6 — Constraint funnel (waterfall)
# ===========================================================================
def fig6_constraint_funnel(features: pd.DataFrame) -> None:
    """
    Waterfall / funnel showing how successive constraints reduce the candidate
    pool from 2860 down to 5 guides.
    """
    n_total = len(features)

    after_gc  = features["gc_content"].between(GC_LOW, GC_HIGH)
    after_tm  = after_gc  & features["melting_temp_c"].between(TM_LOW, TM_HIGH)
    after_ot  = after_tm  & (features["off_target_count"] <= OT_MAX)
    after_cap = after_ot  & ~features["off_target_capped"].astype(bool)

    stages = [
        ("All candidates",                  n_total),
        (f"GC in [{GC_LOW:.2f},{GC_HIGH:.2f}]", after_gc.sum()),
        (f"Tm in [{TM_LOW:.0f},{TM_HIGH:.0f}] C",  after_tm.sum()),
        (f"OT count <= {OT_MAX}",                   after_ot.sum()),
        ("Not OT-capped",                           after_cap.sum()),
    ]

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    colors = [
        "#4361EE",  # all
        "#7209B7",  # after GC
        "#F72585",  # after Tm
        "#4CC9F0",  # after OT
        "#57CC99",  # final feasible
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.55)

    # Reduction arrows
    for i in range(len(counts) - 1):
        removed = counts[i] - counts[i + 1]
        ax.annotate(
            f"-{removed:,}",
            xy=((i + i + 1) / 2, max(counts[i], counts[i + 1]) / 2),
            ha="center", va="center", fontsize=9, color="#888",
            style="italic",
        )

    ax.bar_label(bars, labels=[f"{c:,}" for c in counts],
                 padding=4, fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of guides remaining", **FONT_LABEL)
    ax.set_title("Fig 6 — Biological Constraint Funnel: 2860 → 5 Feasible Guides",
                 **FONT_TITLE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, n_total * 1.15)
    ax.tick_params(axis="x", labelsize=9, rotation=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "fig6_constraint_funnel.png")


# ===========================================================================
# Summary CSV
# ===========================================================================
def write_summary_csv(convex: pd.DataFrame, features: pd.DataFrame) -> None:
    """
    Merge LP output with raw feature columns (GC, Tm, OT count)
    and write results/convex_optimization_summary.csv.
    """
    feat_cols = features[["guide_rna", "gc_content", "melting_temp_c",
                           "off_target_count"]].copy()
    summary = convex.merge(feat_cols, on="guide_rna", how="left", suffixes=("", "_feat"))

    # Prefer columns already in convex if they exist
    for col in ["gc_content", "melting_temp_c", "off_target_count"]:
        if f"{col}_feat" in summary.columns:
            summary[col] = summary[col].combine_first(summary.pop(f"{col}_feat"))

    keep = [
        "rank",
        "guide_rna",
        "objective_coefficient",
        "efficiency_score",
        "specificity_score",
        "structural_suitability_score",
        "gc_content",
        "melting_temp_c",
        "off_target_count",
        "selection_variable",
    ]
    # Add off_target_capped if present
    if "off_target_capped" in summary.columns:
        keep.insert(keep.index("off_target_count") + 1, "off_target_capped")

    available = [c for c in keep if c in summary.columns]
    out = summary[available].sort_values("rank").reset_index(drop=True)

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SUMMARY, index=False)
    log.info("Saved summary CSV to %s  (%d rows, %d cols)", OUT_SUMMARY,
             len(out), len(out.columns))


# ===========================================================================
# Orchestration
# ===========================================================================
def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading inputs ...")
    weighted  = pd.read_csv(IN_WEIGHTED)
    convex    = pd.read_csv(IN_CONVEX)
    features  = pd.read_csv(IN_FEATURES)

    log.info("Loaded: weighted=%d, convex=%d, features=%d rows",
             len(weighted), len(convex), len(features))

    log.info("Generating Fig 1 — objective coefficients ...")
    fig1_objective_coefficients(convex)

    log.info("Generating Fig 2 — constraint breakdown ...")
    fig2_constraint_breakdown(features)

    log.info("Generating Fig 3 — efficiency vs specificity scatter ...")
    fig3_efficiency_vs_specificity(features, convex, weighted)

    log.info("Generating Fig 4 — weighted vs convex comparison ...")
    fig4_weighted_vs_convex(weighted, convex, features)

    log.info("Generating Fig 5 — objective score breakdown ...")
    fig5_objective_breakdown(convex)

    log.info("Generating Fig 6 — constraint funnel ...")
    fig6_constraint_funnel(features)

    log.info("Writing summary CSV ...")
    write_summary_csv(convex, features)

    log.info("All outputs written to %s/", RESULTS_DIR)
    print("\n" + "=" * 60)
    print("Convex Visualization — completed successfully")
    print("=" * 60)
    print(f"  Figures  : {RESULTS_DIR}/")
    print(f"  Summary  : {OUT_SUMMARY}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
