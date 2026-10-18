"""
src/optimization.py

Module 4 of the Multi-Objective CRISPR Guide RNA Design Framework.

Consumes the feature table produced by Module 3 (feature_extraction.py +
Bowtie specificity) and produces a ranked list of guide RNA candidates
via a configurable weighted-sum multi-objective score.

Objectives:
    - efficiency              : Doench 2016-style on-target efficiency proxy
    - specificity              : off-target burden (count + mismatch divergence)
    - structural_suitability   : Tm (target-range desirability) + duplex dG +
                                  guide self-folding energy (MFE)

Input : data/gRNA_features_specificity.csv
Output: data/optimized_guides.csv

Design notes:
    - Weighted-sum scoring is used now (default weights configurable via CLI
      or OptimizationConfig). compute_objective_scores() is kept separate
      from weighted_sum_rank() specifically so a Pareto-front ranking method
      can be added later without touching the normalization/scoring logic.

Author: Bioinformatics Development Team
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimization")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationConfig:
    input_csv: str = "data/gRNA_features_specificity.csv"
    output_csv: str = "data/optimized_guides.csv"

    weight_efficiency: float = 0.35
    weight_specificity: float = 0.40
    weight_structural: float = 0.25

    # Tm target-range desirability window (deg C)
    tm_low: float = 55.0
    tm_high: float = 65.0
    tm_tolerance: float = 5.0

    def validate(self) -> None:
        total = self.weight_efficiency + self.weight_specificity + self.weight_structural
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Weights must sum to 1.0 (got {total:.4f})")


# --------------------------------------------------------------------------- #
# Normalization primitives
# --------------------------------------------------------------------------- #
def min_max_normalize(series: pd.Series, maximize: bool) -> pd.Series:
    """
    Scale a feature to [0, 1]. Direction is explicit: `maximize=True` means
    higher raw values map to scores closer to 1; `maximize=False` inverts it.
    Constant-valued columns return a neutral 0.5 for every row (avoids /0).
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    scaled = (series - lo) / (hi - lo)
    return scaled if maximize else 1.0 - scaled


def target_range_desirability(
    series: pd.Series, low: float, high: float, tolerance: float
) -> pd.Series:
    """
    Derringer-Suich style plateau desirability: score = 1 inside [low, high],
    ramps linearly down to 0 across `tolerance` on either side, 0 beyond that.
    Used for features where "more" is not "better" past a functional optimum
    (e.g. melting temperature).
    """
    def score(x: float) -> float:
        if low <= x <= high:
            return 1.0
        if low - tolerance <= x < low:
            return (x - (low - tolerance)) / tolerance
        if high < x <= high + tolerance:
            return (high + tolerance - x) / tolerance
        return 0.0

    return series.apply(score)


# --------------------------------------------------------------------------- #
# Feature imputation
# --------------------------------------------------------------------------- #
def impute_mismatch_score(df: pd.DataFrame) -> pd.Series:
    """
    mismatch_score is NaN exactly when off_target_count == 0 (nothing to
    average). A guide with zero off-targets is at least as specific as the
    guide with the most divergent observed off-targets, so it is imputed
    with the dataset's max mismatch_score rather than penalized or dropped.
    """
    filled = df["mismatch_score"].copy()
    if filled.notna().any():
        best_case = filled.max()
    else:
        # No guide in the dataset has any off-target hits at all.
        best_case = 1.0
    return filled.fillna(best_case)


# --------------------------------------------------------------------------- #
# Objective scoring
# --------------------------------------------------------------------------- #
def compute_objective_scores(df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """
    Compute normalized per-feature scores and aggregate them into the three
    objective scores: efficiency, specificity, structural_suitability.
    Returns a new DataFrame with only the added score columns (aligned by index).

    Specificity sub-weights (explicit, sum to 1.0):
        70 % off-target burden  (off_target_count_norm, lower count → higher score)
        30 % mismatch divergence (mismatch_score_norm, higher divergence → safer guide)

    Capped guides (off_target_capped == True) were cut off by Bowtie's -k 1000
    limit; their true off-target count is unknown but ≥ 999.  To avoid treating
    the reporting cap as an exact count, capped guides are forced to the worst
    observable off-target burden (score = 0.0) regardless of their raw count.
    """
    scores = pd.DataFrame(index=df.index)

    # --- Efficiency: maximize ---
    scores["efficiency_norm"] = min_max_normalize(df["doench2016_proxy_score"], maximize=True)
    scores["efficiency_score"] = scores["efficiency_norm"]

    # -----------------------------------------------------------------------
    # --- Specificity ---
    #
    # Step 1: off-target burden (70 % of specificity score)
    #
    # Capped guides have off_target_capped == True, meaning Bowtie hit its
    # -k 1000 reporting limit.  The count of 999 is a lower bound, not an
    # exact value.  We therefore assign those guides the worst possible raw
    # count (the dataset maximum) before normalization so that min-max scaling
    # places them at 0.0 on the burden score regardless of other guides' counts.
    # Uncapped guides are normalized using their actual off_target_count.
    # -----------------------------------------------------------------------
    raw_counts = df["off_target_count"].copy()
    if "off_target_capped" in df.columns:
        # Force capped guides to the dataset maximum (or 999 if all are capped).
        dataset_max = raw_counts.max()
        raw_counts = raw_counts.where(~df["off_target_capped"].astype(bool), other=dataset_max)

    scores["off_target_count_norm"] = min_max_normalize(raw_counts, maximize=False)

    # Apply worst-burden score (0.0) to capped guides after normalization as
    # a hard floor, ensuring no floating-point rounding grants them any credit.
    if "off_target_capped" in df.columns:
        capped_mask = df["off_target_capped"].astype(bool)
        scores.loc[capped_mask, "off_target_count_norm"] = 0.0

    # Step 2: mismatch divergence (30 % of specificity score)
    # Higher mean edit distance across off-targets → safer guide → maximize.
    mismatch_filled = impute_mismatch_score(df)
    scores["mismatch_score_norm"] = min_max_normalize(mismatch_filled, maximize=True)

    # Step 3: weighted combination — 70 % burden, 30 % divergence.
    _W_BURDEN = 0.70
    _W_DIVERGENCE = 0.30
    scores["specificity_score"] = (
        _W_BURDEN    * scores["off_target_count_norm"]
        + _W_DIVERGENCE * scores["mismatch_score_norm"]
    )

    # --- Structural suitability: Tm (target-range) + dG (minimize) + MFE (maximize) ---
    scores["melting_temp_norm"] = target_range_desirability(
        df["melting_temp_c"], config.tm_low, config.tm_high, config.tm_tolerance
    )
    scores["delta_g_norm"] = min_max_normalize(df["delta_g_kcal_mol"], maximize=False)
    scores["folding_energy_norm"] = min_max_normalize(df["folding_energy_kcal_mol"], maximize=True)
    scores["structural_suitability_score"] = scores[
        ["melting_temp_norm", "delta_g_norm", "folding_energy_norm"]
    ].mean(axis=1)

    return scores


# --------------------------------------------------------------------------- #
# Weighted-sum ranking (extensible: swap/extend with a Pareto method later)
# --------------------------------------------------------------------------- #
def weighted_sum_rank(scores: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """Combine the three objective scores into a single weighted final_score and rank."""
    result = scores.copy()
    result["final_score"] = (
        config.weight_efficiency * result["efficiency_score"]
        + config.weight_specificity * result["specificity_score"]
        + config.weight_structural * result["structural_suitability_score"]
    )
    result["rank"] = result["final_score"].rank(ascending=False, method="min").astype(int)
    return result.sort_values("rank").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
OUTPUT_COLUMNS = [
    "guide_rna", "pam_sequence", "pam_position", "strand",
    "efficiency_norm",
    "off_target_count_norm", "off_target_capped", "mismatch_score_norm", "specificity_score",
    "melting_temp_norm", "delta_g_norm", "folding_energy_norm", "structural_suitability_score",
    "efficiency_score", "final_score", "rank",
]


def run_pipeline(config: OptimizationConfig) -> pd.DataFrame:
    config.validate()
    log.info("Loading feature table from %s", config.input_csv)
    df = pd.read_csv(config.input_csv)
    if df.empty:
        raise ValueError(f"No guides found in {config.input_csv}")

    required = {
        "guide_rna", "doench2016_proxy_score", "off_target_count", "mismatch_score",
        "melting_temp_c", "delta_g_kcal_mol", "folding_energy_kcal_mol",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    log.info("Scoring %d guides ...", len(df))
    scores = compute_objective_scores(df, config)
    combined = pd.concat([df, scores], axis=1)
    ranked = weighted_sum_rank(combined, config)

    available_cols = [c for c in OUTPUT_COLUMNS if c in ranked.columns]
    output_df = ranked[available_cols]

    Path(config.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config.output_csv, index=False)
    log.info("Saved ranked guide list (%d rows) to %s", len(output_df), config.output_csv)

    log.info(
        "Top guide: %s | final_score=%.4f | efficiency=%.3f | specificity=%.3f | structural=%.3f",
        output_df.iloc[0]["guide_rna"],
        output_df.iloc[0]["final_score"],
        output_df.iloc[0]["efficiency_score"],
        output_df.iloc[0]["specificity_score"],
        output_df.iloc[0]["structural_suitability_score"],
    )
    return output_df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(description="Multi-objective CRISPR guide RNA ranking.")
    parser.add_argument("--input", default="data/gRNA_features_specificity.csv")
    parser.add_argument("--output", default="data/optimized_guides.csv")
    parser.add_argument("--w-efficiency", type=float, default=0.35)
    parser.add_argument("--w-specificity", type=float, default=0.40)
    parser.add_argument("--w-structural", type=float, default=0.25)
    parser.add_argument("--tm-low", type=float, default=55.0)
    parser.add_argument("--tm-high", type=float, default=65.0)
    parser.add_argument("--tm-tolerance", type=float, default=5.0)
    args = parser.parse_args()
    return OptimizationConfig(
        input_csv=args.input,
        output_csv=args.output,
        weight_efficiency=args.w_efficiency,
        weight_specificity=args.w_specificity,
        weight_structural=args.w_structural,
        tm_low=args.tm_low,
        tm_high=args.tm_high,
        tm_tolerance=args.tm_tolerance,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())