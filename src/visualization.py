"""
src/visualization.py

Module 5 of the Multi-Objective CRISPR Guide RNA Design Framework.

Reads the ranked output of Module 4 (optimization.py) and produces static
plots + a concise top-10 summary table. Read-only with respect to its input:
data/optimized_guides.csv is never modified.

Input : data/optimized_guides.csv
Output:
    results/figures/top10_ranked_guides.png
    results/figures/final_score_distribution.png
    results/figures/efficiency_vs_specificity.png
    results/figures/specificity_vs_structural.png
    results/figures/top10_objective_comparison.png
    results/optimization_summary.csv

Author: Bioinformatics Development Team
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless/non-interactive backend, safe for scripts & CI
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("visualization")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

TOP_N = 10


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class VisualizationConfig:
    input_csv: str = "data/optimized_guides.csv"
    figures_dir: str = "results/figures"
    summary_csv: str = "results/optimization_summary.csv"
    top_n: int = TOP_N


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_ranked_guides(input_csv: str) -> pd.DataFrame:
    """Load the optimized/ranked guide table. Never mutates the source file."""
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Ranked guide CSV not found: {input_csv}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No guides found in {input_csv}")
    return df.sort_values("rank").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. Top 10 ranked gRNAs with final scores
# --------------------------------------------------------------------------- #
def plot_top10_final_scores(df: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    top = df.head(top_n)
    labels = [f"#{r} {g[:10]}…" for r, g in zip(top["rank"], top["guide_rna"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels[::-1], top["final_score"][::-1], color="#2f6f9f")
    ax.set_xlabel("Final weighted score")
    ax.set_title(f"Top {top_n} Ranked Guide RNAs")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, max(1.0, top["final_score"].max() * 1.15))

    out_path = out_dir / "top10_ranked_guides.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 2. Distribution of final scores
# --------------------------------------------------------------------------- #
def plot_final_score_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["final_score"], bins=min(30, max(5, len(df) // 5)), color="#5b8c5a", edgecolor="white")
    ax.axvline(df["final_score"].mean(), color="#c0392b", linestyle="--", label=f"mean = {df['final_score'].mean():.3f}")
    ax.set_xlabel("Final weighted score")
    ax.set_ylabel("Number of guides")
    ax.set_title(f"Distribution of Final Scores (n={len(df)})")
    ax.legend()

    out_path = out_dir / "final_score_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 3 & 4. Objective-pair scatter plots
# --------------------------------------------------------------------------- #
def plot_objective_scatter(
    df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str,
    title: str, filename: str, out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter(
        df[x_col], df[y_col], c=df["final_score"], cmap="viridis",
        s=25, alpha=0.75, edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Final score")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    out_path = out_dir / filename
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 5. Top 10 guides: efficiency vs specificity vs structural suitability
# --------------------------------------------------------------------------- #
def plot_top10_objective_comparison(df: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    top = df.head(top_n)
    labels = [f"#{r}" for r in top["rank"]]
    objectives = {
        "Efficiency": top["efficiency_score"],
        "Specificity": top["specificity_score"],
        "Structural suitability": top["structural_suitability_score"],
    }

    x = range(len(top))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (name, values) in enumerate(objectives.items()):
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], values, width=width, label=name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Normalized objective score")
    ax.set_title(f"Top {top_n} Guides — Objective Breakdown")
    ax.legend()

    out_path = out_dir / "top10_objective_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 6. Summary statistics
# --------------------------------------------------------------------------- #
def compute_summary_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total_guides": len(df),
        "top_score": round(df["final_score"].max(), 4),
        "top_guide": df.loc[df["final_score"].idxmax(), "guide_rna"],
        "mean_final_score": round(df["final_score"].mean(), 4),
        "median_final_score": round(df["final_score"].median(), 4),
    }
    if "off_target_capped" in df.columns:
        stats["capped_guides"] = int(df["off_target_capped"].astype(bool).sum())
    else:
        stats["capped_guides"] = "N/A (off_target_capped not present in input)"
    return stats


def print_summary(stats: dict) -> None:
    log.info("=== Summary Statistics ===")
    for key, value in stats.items():
        log.info("  %-20s : %s", key, value)


# --------------------------------------------------------------------------- #
# Top-10 summary table
# --------------------------------------------------------------------------- #
SUMMARY_COLUMNS = [
    "rank", "guide_rna", "pam_sequence", "pam_position", "strand",
    "efficiency_score", "specificity_score", "structural_suitability_score",
    "final_score",
]


def save_top10_summary(df: pd.DataFrame, output_csv: str, top_n: int) -> pd.DataFrame:
    top = df.head(top_n)
    cols = [c for c in SUMMARY_COLUMNS if c in top.columns]
    summary = top[cols]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    log.info("Saved top-%d summary table to %s", top_n, out_path)
    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(config: VisualizationConfig) -> dict:
    df = load_ranked_guides(config.input_csv)
    log.info("Loaded %d ranked guides from %s", len(df), config.input_csv)

    figures_dir = Path(config.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated = {
        "top10_bar": plot_top10_final_scores(df, figures_dir, config.top_n),
        "score_distribution": plot_final_score_distribution(df, figures_dir),
        "efficiency_vs_specificity": plot_objective_scatter(
            df, "efficiency_score", "specificity_score",
            "Efficiency score", "Specificity score",
            "Efficiency vs Specificity", "efficiency_vs_specificity.png", figures_dir,
        ),
        "specificity_vs_structural": plot_objective_scatter(
            df, "specificity_score", "structural_suitability_score",
            "Specificity score", "Structural suitability score",
            "Specificity vs Structural Suitability", "specificity_vs_structural.png", figures_dir,
        ),
        "top10_objective_comparison": plot_top10_objective_comparison(df, figures_dir, config.top_n),
    }
    for name, path in generated.items():
        log.info("Saved figure [%s] -> %s", name, path)

    stats = compute_summary_stats(df)
    print_summary(stats)

    save_top10_summary(df, config.summary_csv, config.top_n)

    return {"figures": generated, "stats": stats}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> VisualizationConfig:
    parser = argparse.ArgumentParser(description="Visualize and summarize ranked CRISPR guide RNAs.")
    parser.add_argument("--input", default="data/optimized_guides.csv")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--summary-csv", default="results/optimization_summary.csv")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()
    return VisualizationConfig(
        input_csv=args.input,
        figures_dir=args.figures_dir,
        summary_csv=args.summary_csv,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())