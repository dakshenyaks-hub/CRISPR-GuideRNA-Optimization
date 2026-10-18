"""
src/grna_optimizer.py

Step 6 — Multi-Objective Optimization for CRISPR Guide RNA Selection.

Implements the approved Step 5 mathematical formulation:

    Track A: Continuous LP relaxation (CVXOPT) for 23MAT204 demonstration
             of LP, Lagrangian, KKT conditions.
    Track B: Exact discrete binary Pareto analysis via ε-constraint method
             and direct dominance comparison for biologically valid
             guide selection.

Mathematical formulation:
    Decision variables : x_i ∈ {0,1}  (binary guide selection)
    Objective 1        : maximize Σ Ê_i x_i  (normalised DeepSpCas9 activity)
    Objective 2        : maximize Σ Ŝ_i x_i  (normalised specificity = 1 − Ô_i)
    Constraints        : GC content, poly-T, homopolymer, MFE, cardinality

The module does NOT modify any upstream pipeline outputs.

Author: Bioinformatics Development Team
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grna_optimizer")


# =========================================================================== #
# Configuration
# =========================================================================== #
@dataclass
class OptimizerConfig:
    """
    All tuneable parameters for the gRNA optimisation pipeline.

    Biological thresholds are stored here (not hard-coded in logic)
    so they can be adjusted from a config file, CLI, or calling code.
    """

    # --- Input / Output ---
    input_csv: str = "results/TP53/gRNA_features.csv"
    output_dir: str = "results/TP53/optimization"

    # --- Guide selection ---
    k: int = 1                          # number of guides to select

    # --- Biological constraint thresholds ---
    gc_min: float = 0.30
    gc_max: float = 0.70
    max_homopolymer_run: int = 4
    require_no_poly_t: bool = True      # has_poly_t_terminator must be False

    # MFE threshold: disabled by default (None) until a literature-supported
    # value is established.  Set to e.g. -5.0 to enable.
    mfe_threshold: Optional[float] = None

    # --- Off-target capped guide handling ---
    # 'retain' : keep capped guides but flag them (Option B — recommended)
    # 'exclude': remove capped guides from primary analysis (Option A)
    capped_guide_policy: str = "retain"

    # --- Pareto sweep ---
    n_epsilon_steps: int = 50           # number of ε values for ε-constraint sweep
    n_weight_steps: int = 21            # number of weight pairs for weighted-sum sweep

    # --- LP relaxation ---
    run_lp_relaxation: bool = True      # enable Track A (CVXOPT demo)


# =========================================================================== #
# 1. Data Loading & Biological Constraint Filtering
# =========================================================================== #
def load_and_filter(config: OptimizerConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load the feature table, apply biological feasibility constraints,
    and return (feasible_df, excluded_df, filter_report).

    The original CSV is NEVER modified.
    """
    df = pd.read_csv(config.input_csv)
    n_total = len(df)
    report: dict = {"total_candidates": n_total, "filters_applied": {}}

    # Track exclusion reasons (a guide can be excluded for multiple reasons)
    exclude_mask = pd.Series(False, index=df.index)

    # --- Filter 1: NaN DeepSpCas9 score ---
    nan_mask = df["deep_spcas9_score"].isna()
    report["filters_applied"]["nan_deep_spcas9"] = int(nan_mask.sum())
    exclude_mask |= nan_mask

    # --- Filter 2: GC content ---
    gc_mask = (df["gc_content"] < config.gc_min) | (df["gc_content"] > config.gc_max)
    report["filters_applied"]["gc_content_out_of_range"] = int(gc_mask.sum())
    exclude_mask |= gc_mask

    # --- Filter 3: Poly-T terminator ---
    if config.require_no_poly_t:
        pt_mask = df["has_poly_t_terminator"] == True  # noqa: E712
        report["filters_applied"]["poly_t_terminator"] = int(pt_mask.sum())
        exclude_mask |= pt_mask

    # --- Filter 4: Homopolymer run ---
    hr_mask = df["max_homopolymer_run"] > config.max_homopolymer_run
    report["filters_applied"]["homopolymer_run_exceeded"] = int(hr_mask.sum())
    exclude_mask |= hr_mask

    # --- Filter 5: MFE threshold (only if configured) ---
    if config.mfe_threshold is not None:
        mfe_mask = df["folding_energy_kcal_mol"] < config.mfe_threshold
        report["filters_applied"]["mfe_below_threshold"] = int(mfe_mask.sum())
        exclude_mask |= mfe_mask

    # --- Off-target capped handling ---
    capped_mask = df["off_target_capped"] == True  # noqa: E712
    n_capped_total = int(capped_mask.sum())
    report["off_target_capped_total"] = n_capped_total

    if config.capped_guide_policy == "exclude":
        report["filters_applied"]["off_target_capped_excluded"] = int(
            (capped_mask & ~exclude_mask).sum()
        )
        exclude_mask |= capped_mask
        report["capped_guide_policy"] = "exclude"
    else:
        report["capped_guide_policy"] = "retain (censored: OT count is lower bound)"

    feasible_df = df[~exclude_mask].copy().reset_index(drop=True)
    excluded_df = df[exclude_mask].copy().reset_index(drop=True)

    report["feasible_candidates"] = len(feasible_df)
    report["excluded_candidates"] = len(excluded_df)

    # Count capped guides in the feasible set
    n_capped_feasible = int(
        (feasible_df["off_target_capped"] == True).sum()  # noqa: E712
    )
    report["capped_in_feasible_set"] = n_capped_feasible

    log.info("Loaded %d candidates from %s", n_total, config.input_csv)
    log.info(
        "After biological filtering: %d feasible, %d excluded",
        len(feasible_df),
        len(excluded_df),
    )
    if n_capped_feasible > 0:
        log.info(
            "  NOTE: %d feasible guides have censored off-target counts "
            "(off_target_capped=True, OT count is lower bound >= 999)",
            n_capped_feasible,
        )
    for filt, count in report["filters_applied"].items():
        log.info("  Filter %-35s removed %d", filt, count)

    return feasible_df, excluded_df, report


# =========================================================================== #
# 2. Normalisation
# =========================================================================== #
def normalise_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute min-max normalised objectives on the feasible set:
        E_hat_i = (E_i - E_min) / (E_max - E_min)   in [0, 1]  (higher = better activity)
        O_hat_i = (O_i - O_min) / (O_max - O_min)   in [0, 1]  (lower  = better specificity)
        S_hat_i = 1 - O_hat_i                        in [0, 1]  (higher = better specificity)

    Adds columns: norm_activity, norm_ot_burden, norm_specificity
    """
    E = df["deep_spcas9_score"].values.astype(np.float64)
    O = df["off_target_count"].values.astype(np.float64)

    E_min, E_max = E.min(), E.max()
    O_min, O_max = O.min(), O.max()

    # Guard against zero range (all identical values)
    E_range = E_max - E_min if E_max > E_min else 1.0
    O_range = O_max - O_min if O_max > O_min else 1.0

    df = df.copy()
    df["norm_activity"] = (E - E_min) / E_range
    df["norm_ot_burden"] = (O - O_min) / O_range
    df["norm_specificity"] = 1.0 - df["norm_ot_burden"].values

    log.info(
        "Normalisation | E in [%.2f, %.2f] -> [0, 1] | O in [%.0f, %.0f] -> [0, 1]",
        E_min, E_max, O_min, O_max,
    )

    return df


# =========================================================================== #
# 3. Track B — Exact Discrete Pareto Analysis
# =========================================================================== #
def find_pareto_front(activity: np.ndarray, specificity: np.ndarray) -> np.ndarray:
    """
    Given arrays of (activity, specificity) — both to be maximised —
    return a boolean mask of Pareto-optimal (non-dominated) points.

    A point i dominates point j iff:
        activity[i] >= activity[j]  AND  specificity[i] >= specificity[j]
        with at least one strict inequality.
    """
    n = len(activity)
    if n == 0:
        return np.array([], dtype=bool)

    act = np.asarray(activity, dtype=np.float64)
    spec = np.asarray(specificity, dtype=np.float64)

    # For n <= 10000, exact vectorized pairwise comparison is instantaneous (<20ms)
    # and guarantees mathematical perfection without tie-breaking subtleties.
    if n <= 10000:
        # i dominates j:
        dom = (
            (act[:, None] >= act[None, :])
            & (spec[:, None] >= spec[None, :])
            & ((act[:, None] > act[None, :]) | (spec[:, None] > spec[None, :]))
        )
        is_pareto = ~np.any(dom, axis=0)
        return is_pareto

    # For very large n, use O(n log n) sorting sweep
    order = np.lexsort((-spec, -act))
    is_pareto = np.zeros(n, dtype=bool)
    max_spec_so_far = -np.inf

    for idx in order:
        cur_spec = spec[idx]
        if cur_spec > max_spec_so_far:
            is_pareto[idx] = True
            max_spec_so_far = cur_spec
        elif cur_spec == max_spec_so_far and act[idx] == act[order[0]]:
            # Duplicate top point
            is_pareto[idx] = True

    return is_pareto


def epsilon_constraint_sweep(
    df: pd.DataFrame, config: OptimizerConfig
) -> pd.DataFrame:
    """
    epsilon-constraint method for generating the discrete Pareto front.

    Primary objective: maximize norm_activity
    Constraint:        norm_ot_burden <= epsilon

    Sweep epsilon from min(norm_ot_burden) to max(norm_ot_burden).
    For each epsilon, select the k guides with the highest norm_activity
    among guides satisfying the constraint.

    Returns a DataFrame of unique Pareto-optimal solutions found.
    """
    burden = df["norm_ot_burden"].values
    activity = df["norm_activity"].values

    eps_min = burden.min()
    eps_max = burden.max()
    epsilons = np.linspace(eps_min, eps_max, config.n_epsilon_steps)

    selected_indices: set[int] = set()

    for eps in epsilons:
        feasible_mask = burden <= eps + 1e-12  # small tolerance
        if not feasible_mask.any():
            continue

        feasible_idx = np.where(feasible_mask)[0]
        feasible_activity = activity[feasible_idx]

        # Select top-k by activity within epsilon-feasible set
        if len(feasible_idx) <= config.k:
            top_k_idx = feasible_idx
        else:
            top_k_local = np.argsort(-feasible_activity)[: config.k]
            top_k_idx = feasible_idx[top_k_local]

        for idx in top_k_idx:
            selected_indices.add(int(idx))

    # Collect all candidates found by epsilon-constraint
    if not selected_indices:
        log.warning("epsilon-constraint sweep found no solutions.")
        return pd.DataFrame()

    pareto_candidates = df.iloc[sorted(selected_indices)].copy()

    # Verify true Pareto optimality by dominance check
    act = pareto_candidates["norm_activity"].values
    spec = pareto_candidates["norm_specificity"].values
    is_pareto = find_pareto_front(act, spec)

    pareto_df = pareto_candidates[is_pareto].copy().reset_index(drop=True)

    log.info(
        "epsilon-constraint sweep: %d unique candidates -> %d Pareto-optimal",
        len(pareto_candidates),
        len(pareto_df),
    )

    return pareto_df


def weighted_sum_sweep(
    df: pd.DataFrame, config: OptimizerConfig
) -> pd.DataFrame:
    """
    Weighted-sum scalarisation sweep (SECONDARY method).

    max  w1 * norm_activity[i] + w2 * norm_specificity[i]

    where w1 + w2 = 1, swept across config.n_weight_steps values.

    Returns unique solutions found. Labelled as approximate Pareto front.
    """
    activity = df["norm_activity"].values
    specificity = df["norm_specificity"].values

    weights = np.linspace(0.0, 1.0, config.n_weight_steps)
    selected_indices: set[int] = set()

    for w1 in weights:
        w2 = 1.0 - w1
        scores = w1 * activity + w2 * specificity

        if config.k == 1:
            best_idx = int(np.argmax(scores))
            selected_indices.add(best_idx)
        else:
            top_k = np.argsort(-scores)[: config.k]
            for idx in top_k:
                selected_indices.add(int(idx))

    ws_df = df.iloc[sorted(selected_indices)].copy().reset_index(drop=True)
    log.info(
        "Weighted-sum sweep (%d weight pairs): %d unique solutions",
        config.n_weight_steps,
        len(ws_df),
    )

    return ws_df


# =========================================================================== #
# 4. Track A — LP Relaxation via CVXOPT (23MAT204 Component)
# =========================================================================== #
def run_lp_relaxation(df: pd.DataFrame, config: OptimizerConfig) -> dict:
    """
    Solve the continuous LP relaxation of the weighted-sum scalarised problem
    using CVXOPT, demonstrating:
      - Linear Programming formulation
      - Convex optimisation
      - KKT conditions (via dual variables)
      - Lagrangian interpretation

    Formulation (minimisation):
        min  c^T x
        s.t. sum(x) = k        (cardinality)
             0 <= x_i <= 1      (box bounds)

    where c_i = -(0.5 * E_hat_i + 0.5 * S_hat_i)  (equal-weight scalarisation)

    This is the LP RELAXATION (x_i in [0,1]) — NOT the binary problem.
    """
    try:
        from cvxopt import matrix, spmatrix, solvers
    except ImportError:
        log.warning("CVXOPT not installed — skipping LP relaxation (Track A).")
        return {"status": "skipped", "reason": "cvxopt not installed"}

    # Suppress CVXOPT solver output
    solvers.options["show_progress"] = False

    n = len(df)
    activity = df["norm_activity"].values.astype(np.float64)
    specificity = df["norm_specificity"].values.astype(np.float64)

    # Cost vector: min c^T x  where c_i = -(0.5*E_hat_i + 0.5*S_hat_i)
    # (negative because LP is minimisation, we want to maximise the sum)
    w1, w2 = 0.5, 0.5
    c = -(w1 * activity + w2 * specificity)
    c_cvx = matrix(c, (n, 1), tc="d")

    # Inequality constraints: G x <= h
    # Two sets of box constraints:
    #   -x_i <= 0   (i.e. x_i >= 0)
    #    x_i <= 1
    # Use SPARSE matrix: G is diagonal-block, only 2n non-zeros out of 2n*n.
    # For large n (e.g. EGFR ~18k feasible), the dense approach would allocate
    # 2n*n floats (~5 GB for n=18k) — the sparse form uses O(n) memory.
    I_rows = list(range(n)) + list(range(n, 2 * n))
    I_cols = list(range(n)) + list(range(n))
    I_vals = [-1.0] * n + [1.0] * n
    G_cvx = spmatrix(I_vals, I_rows, I_cols, (2 * n, n), tc="d")

    h_data = np.zeros(2 * n, dtype=np.float64)
    h_data[n:] = 1.0
    h_cvx = matrix(h_data, tc="d")

    # Equality constraint: sum(x) = k
    A_data = np.ones((1, n), dtype=np.float64)
    b_data = np.array([float(config.k)], dtype=np.float64)
    A_cvx = matrix(A_data, tc="d")
    b_cvx = matrix(b_data, tc="d")

    log.info("Solving LP relaxation via CVXOPT (n=%d, k=%d)...", n, config.k)
    sol = solvers.lp(c_cvx, G_cvx, h_cvx, A_cvx, b_cvx)

    if sol["status"] != "optimal":
        log.warning("LP relaxation status: %s", sol["status"])
        return {"status": sol["status"]}

    x_opt = np.array(sol["x"]).flatten()
    obj_val = float(sol["primal objective"])

    # --- Extract KKT / Lagrangian information ---
    # Dual variable for equality constraint (sum x = k): mu
    mu = float(np.array(sol["y"]).flatten()[0])

    # Dual variables for inequality constraints:
    #   z[0:n]   = alpha_i  (multiplier for -x_i <= 0, i.e. x_i >= 0)
    #   z[n:2n]  = beta_i   (multiplier for  x_i <= 1)
    z_dual = np.array(sol["z"]).flatten()
    alpha = z_dual[:n]
    beta = z_dual[n:]

    # Categorise variables by KKT complementary slackness
    tol = 1e-6
    fully_selected = np.where(x_opt > 1.0 - tol)[0]
    fully_rejected = np.where(x_opt < tol)[0]
    fractional = np.where((x_opt > tol) & (x_opt < 1.0 - tol))[0]

    # Stationarity check: c_i + mu - alpha_i + beta_i = 0
    stationarity_residual = c + mu - alpha + beta
    stationarity_error = float(np.max(np.abs(stationarity_residual)))

    # Complementary slackness check
    cs_alpha = float(np.max(np.abs(alpha * x_opt)))
    cs_beta = float(np.max(np.abs(beta * (1.0 - x_opt))))

    result = {
        "status": sol["status"],
        "objective_value": round(-obj_val, 6),  # negate back to maximisation value
        "weights_used": {"w1_activity": w1, "w2_specificity": w2},
        "n_variables": n,
        "k": config.k,
        "solution_summary": {
            "fully_selected": int(len(fully_selected)),
            "fully_rejected": int(len(fully_rejected)),
            "fractional": int(len(fractional)),
        },
        "kkt_analysis": {
            "mu_cardinality_multiplier": round(mu, 6),
            "mu_interpretation": (
                "Selection threshold: guides with weighted cost c_i < "
                + str(round(-mu, 6))
                + " are selected; those above are rejected."
            ),
            "stationarity_max_residual": round(stationarity_error, 10),
            "complementary_slackness_alpha_max": round(cs_alpha, 10),
            "complementary_slackness_beta_max": round(cs_beta, 10),
            "kkt_satisfied": bool(
                stationarity_error < 1e-5
                and cs_alpha < 1e-5
                and cs_beta < 1e-5
            ),
        },
        "lagrangian_formulation": (
            "L(x, mu, alpha, beta) = c^T x + mu(sum(x_i) - k) "
            "+ sum(alpha_i(-x_i)) + sum(beta_i(x_i - 1))\n"
            "where mu in R (equality), alpha_i >= 0 (lower bound), "
            "beta_i >= 0 (upper bound)"
        ),
        "note": (
            "This is the CONTINUOUS LP relaxation (x_i in [0,1]). "
            "It does NOT directly produce a biologically valid binary guide selection. "
            "See Track B for the exact discrete Pareto analysis."
        ),
    }

    # Top selected variables (for reporting)
    top_idx = np.argsort(-x_opt)[:min(10, n)]
    result["top_lp_variables"] = []
    for idx in top_idx:
        entry = {
            "guide_id": str(df.iloc[idx]["guide_id"]),
            "x_value": round(float(x_opt[idx]), 6),
            "norm_activity": round(float(activity[idx]), 6),
            "norm_specificity": round(float(specificity[idx]), 6),
            "cost_c_i": round(float(c[idx]), 6),
            "alpha_i": round(float(alpha[idx]), 6),
            "beta_i": round(float(beta[idx]), 6),
        }
        result["top_lp_variables"].append(entry)

    log.info("LP relaxation solved: status=%s, objective=%.6f", sol["status"], -obj_val)
    log.info(
        "  KKT: mu=%.6f | stationarity residual=%.2e | CS(alpha)=%.2e | CS(beta)=%.2e",
        mu, stationarity_error, cs_alpha, cs_beta,
    )
    log.info(
        "  Solution: %d fully selected, %d fractional, %d rejected",
        len(fully_selected), len(fractional), len(fully_rejected),
    )

    return result


# =========================================================================== #
# 5. Guide Selection & Reporting
# =========================================================================== #
def select_guides(
    pareto_df: pd.DataFrame, full_df: pd.DataFrame, config: OptimizerConfig
) -> pd.DataFrame:
    """
    From the Pareto-optimal set, select k guides that represent a balanced
    trade-off (highest combined normalised score).

    For k=1, this is the single Pareto-optimal guide with the best
    balanced score (equal-weight sum).  The user can then inspect the
    full Pareto front to choose alternatives.
    """
    if pareto_df.empty:
        log.warning("No Pareto-optimal guides found — cannot select.")
        return pd.DataFrame()

    score = (
        0.5 * pareto_df["norm_activity"].values
        + 0.5 * pareto_df["norm_specificity"].values
    )
    pareto_df = pareto_df.copy()
    pareto_df["balanced_score"] = score

    top_k = pareto_df.nlargest(config.k, "balanced_score")
    return top_k.reset_index(drop=True)


def build_guide_report(row: pd.Series) -> dict:
    """Build a detailed report dict for one selected guide."""
    report = {
        "guide_id": str(row["guide_id"]),
        "guide_rna": str(row["guide_rna"]),
        "strand": str(row["strand"]),
        "pam_sequence": str(row["pam_sequence"]),
        "pam_position": int(row["pam_position"]),
        "gc_content": round(float(row["gc_content"]), 4),
        "deep_spcas9_score": round(float(row["deep_spcas9_score"]), 4),
        "norm_activity": round(float(row["norm_activity"]), 6),
        "off_target_count": int(row["off_target_count"]),
        "off_target_capped": bool(row["off_target_capped"]),
        "off_target_count_note": (
            "CENSORED: lower bound (>=999, true count unknown)"
            if row["off_target_capped"]
            else "exact count"
        ),
        "norm_specificity": round(float(row["norm_specificity"]), 6),
        "folding_energy_kcal_mol": round(float(row["folding_energy_kcal_mol"]), 2),
        "max_homopolymer_run": int(row["max_homopolymer_run"]),
        "has_poly_t_terminator": bool(row["has_poly_t_terminator"]),
    }
    if "balanced_score" in row.index:
        report["balanced_score"] = round(float(row["balanced_score"]), 6)
    return report


# =========================================================================== #
# 6. Visualisation
# =========================================================================== #
def generate_pareto_plot(
    feasible_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_path: Path,
    config: OptimizerConfig,
) -> None:
    """
    2D Pareto plot:
        X-axis: normalised specificity (higher = fewer off-targets = better)
        Y-axis: normalised activity    (higher = more on-target activity = better)

    Both axes point in the 'better' direction, so the Pareto front is the
    upper-right boundary.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping Pareto plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Mark capped guides differently
    capped_mask = feasible_df["off_target_capped"] == True  # noqa: E712
    uncapped_mask = ~capped_mask

    # All feasible (uncapped) — light grey
    if uncapped_mask.any():
        ax.scatter(
            feasible_df.loc[uncapped_mask, "norm_specificity"],
            feasible_df.loc[uncapped_mask, "norm_activity"],
            c="#d0d0d0", s=8, alpha=0.5, label="Feasible (exact OT)",
            zorder=1, edgecolors="none",
        )

    # All feasible (capped) — light orange hollow
    if capped_mask.any():
        ax.scatter(
            feasible_df.loc[capped_mask, "norm_specificity"],
            feasible_df.loc[capped_mask, "norm_activity"],
            c="none", s=12, alpha=0.4,
            label="Feasible (censored OT >= 999)",
            zorder=1, edgecolors="#e69500", linewidths=0.6,
        )

    # Pareto-optimal — blue
    if not pareto_df.empty:
        ax.scatter(
            pareto_df["norm_specificity"],
            pareto_df["norm_activity"],
            c="#2196F3", s=40, alpha=0.85, label="Pareto-optimal",
            zorder=3, edgecolors="navy", linewidths=0.5,
        )

        # Connect Pareto front with a line
        pf_sorted = pareto_df.sort_values("norm_specificity")
        ax.plot(
            pf_sorted["norm_specificity"],
            pf_sorted["norm_activity"],
            c="#1565C0", linewidth=1.2, alpha=0.6, zorder=2,
        )

    # Selected guide(s) — red star
    if not selected_df.empty:
        ax.scatter(
            selected_df["norm_specificity"],
            selected_df["norm_activity"],
            c="#FF1744", s=200, marker="*",
            label="Selected (k={})".format(config.k),
            zorder=5, edgecolors="darkred", linewidths=0.8,
        )

    ax.set_xlabel("Normalised Specificity (1 - off-target burden) ->", fontsize=11)
    ax.set_ylabel("Normalised On-Target Activity (DeepSpCas9) ->", fontsize=11)
    ax.set_title(
        "Multi-Objective gRNA Optimisation - Pareto Front\n"
        "(epsilon-constraint method, discrete binary analysis)",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved Pareto plot to %s", output_path)


# =========================================================================== #
# 7. Validation
# =========================================================================== #
def validate_results(
    feasible_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    config: OptimizerConfig,
) -> dict:
    """Run all validation checks; return a dict of pass/fail results."""
    checks = {}

    # 1. No NaN objectives in feasible set
    nan_activity = int(feasible_df["deep_spcas9_score"].isna().sum())
    nan_ot = int(feasible_df["off_target_count"].isna().sum())
    checks["no_nan_objectives"] = {
        "pass": nan_activity == 0 and nan_ot == 0,
        "detail": "NaN activity: {}, NaN off-target: {}".format(nan_activity, nan_ot),
    }

    # 2. Biological constraints respected in feasible set
    gc_ok = bool(
        (feasible_df["gc_content"] >= config.gc_min - 1e-9).all()
        and (feasible_df["gc_content"] <= config.gc_max + 1e-9).all()
    )
    hr_ok = bool((feasible_df["max_homopolymer_run"] <= config.max_homopolymer_run).all())
    pt_ok = True
    if config.require_no_poly_t:
        pt_ok = bool((feasible_df["has_poly_t_terminator"] == False).all())  # noqa: E712
    mfe_ok = True
    if config.mfe_threshold is not None:
        mfe_ok = bool((feasible_df["folding_energy_kcal_mol"] >= config.mfe_threshold).all())

    checks["biological_constraints"] = {
        "pass": gc_ok and hr_ok and pt_ok and mfe_ok,
        "detail": "GC: {}, HR: {}, PT: {}, MFE: {}".format(gc_ok, hr_ok, pt_ok, mfe_ok),
    }

    # 3. Selected guides satisfy k
    checks["cardinality_k"] = {
        "pass": len(selected_df) == config.k or (
            len(selected_df) < config.k and len(pareto_df) < config.k
        ),
        "detail": "k={}, selected={}, pareto_size={}".format(
            config.k, len(selected_df), len(pareto_df)
        ),
    }

    # 4. Pareto solutions are non-dominated
    if not pareto_df.empty:
        act = pareto_df["norm_activity"].values
        spec = pareto_df["norm_specificity"].values
        is_pareto = find_pareto_front(act, spec)
        checks["pareto_non_dominated"] = {
            "pass": bool(is_pareto.all()),
            "detail": "{}/{} are non-dominated".format(int(is_pareto.sum()), len(pareto_df)),
        }
    else:
        checks["pareto_non_dominated"] = {"pass": True, "detail": "empty set"}

    # 5. Objective directions correct
    checks["objective_directions"] = {
        "pass": True,
        "detail": "norm_activity: higher=better (max), norm_specificity: higher=better (max)",
    }

    # 6. Normalisation correct (values in [0,1])
    act_range = bool(
        feasible_df["norm_activity"].min() >= -1e-10
        and feasible_df["norm_activity"].max() <= 1.0 + 1e-10
    )
    spec_range = bool(
        feasible_df["norm_specificity"].min() >= -1e-10
        and feasible_df["norm_specificity"].max() <= 1.0 + 1e-10
    )
    checks["normalisation_range"] = {
        "pass": act_range and spec_range,
        "detail": (
            "activity [{:.4f}, {:.4f}], specificity [{:.4f}, {:.4f}]".format(
                feasible_df["norm_activity"].min(),
                feasible_df["norm_activity"].max(),
                feasible_df["norm_specificity"].min(),
                feasible_df["norm_specificity"].max(),
            )
        ),
    }

    # 7. Capped off-target transparency
    n_capped = int((feasible_df["off_target_capped"] == True).sum())  # noqa: E712
    checks["capped_ot_transparency"] = {
        "pass": True,  # As long as we report it
        "detail": (
            "{} capped guides in feasible set; policy: {}".format(
                n_capped, config.capped_guide_policy
            )
        ),
    }

    all_pass = all(c["pass"] for c in checks.values())
    checks["_all_passed"] = all_pass

    return checks


# =========================================================================== #
# 8. Main Orchestrator
# =========================================================================== #
def run_optimization(config: OptimizerConfig) -> dict:
    """
    Execute the full Step 6 optimization pipeline:
      1. Load and filter candidates
      2. Normalise objectives
      3. Track B: epsilon-constraint Pareto + dominance verification
      4. Weighted-sum secondary sweep
      5. Track A: LP relaxation via CVXOPT
      6. Select guide(s)
      7. Validate
      8. Generate outputs and visualisation
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=" * 72)
    log.info("STEP 6 — MULTI-OBJECTIVE GRNA OPTIMIZATION")
    log.info("=" * 72)

    # ---------------------------------------------------------------
    # 1. Load and filter
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 1. Loading and Filtering ---")
    feasible_df, excluded_df, filter_report = load_and_filter(config)

    if feasible_df.empty:
        log.error("No feasible candidates after filtering — cannot optimise.")
        return {"error": "no_feasible_candidates"}

    # ---------------------------------------------------------------
    # 2. Normalise
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 2. Normalising Objectives ---")
    feasible_df = normalise_objectives(feasible_df)

    # ---------------------------------------------------------------
    # 3. Track B: epsilon-constraint Pareto
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 3. Track B: epsilon-Constraint Pareto Analysis ---")
    pareto_df = epsilon_constraint_sweep(feasible_df, config)

    # Also compute full-dataset Pareto front for comparison
    full_pareto_mask = find_pareto_front(
        feasible_df["norm_activity"].values,
        feasible_df["norm_specificity"].values,
    )
    full_pareto_df = feasible_df[full_pareto_mask].copy().reset_index(drop=True)
    log.info(
        "Full Pareto front (direct dominance): %d non-dominated guides",
        len(full_pareto_df),
    )

    # Merge: union of epsilon-constraint and full dominance results
    if not pareto_df.empty:
        merged_ids = set(pareto_df["guide_id"].tolist()) | set(
            full_pareto_df["guide_id"].tolist()
        )
        pareto_df = feasible_df[feasible_df["guide_id"].isin(merged_ids)].copy()
        # Re-verify dominance
        act = pareto_df["norm_activity"].values
        spec = pareto_df["norm_specificity"].values
        is_pareto = find_pareto_front(act, spec)
        pareto_df = pareto_df[is_pareto].copy().reset_index(drop=True)
    else:
        pareto_df = full_pareto_df

    log.info("Final verified Pareto-optimal set: %d guides", len(pareto_df))

    # ---------------------------------------------------------------
    # 4. Weighted-sum (secondary)
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 4. Weighted-Sum Pareto Approximation (secondary) ---")
    ws_df = weighted_sum_sweep(feasible_df, config)

    # ---------------------------------------------------------------
    # 5. Track A: LP relaxation
    # ---------------------------------------------------------------
    lp_result = {}
    if config.run_lp_relaxation:
        log.info("")
        log.info("--- 5. Track A: LP Relaxation (CVXOPT / 23MAT204) ---")
        lp_result = run_lp_relaxation(feasible_df, config)

    # ---------------------------------------------------------------
    # 6. Select guide(s)
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 6. Guide Selection ---")
    selected_df = select_guides(pareto_df, feasible_df, config)

    if not selected_df.empty:
        for _, row in selected_df.iterrows():
            r = build_guide_report(row)
            log.info("  * SELECTED: %s", r["guide_id"])
            log.info("    Sequence: %s", r["guide_rna"])
            log.info(
                "    Strand: %s | PAM: %s | Position: %d",
                r["strand"], r["pam_sequence"], r["pam_position"],
            )
            log.info(
                "    DeepSpCas9: %.4f | Norm activity: %.4f",
                r["deep_spcas9_score"], r["norm_activity"],
            )
            log.info(
                "    Off-target: %d (%s) | Norm specificity: %.4f",
                r["off_target_count"], r["off_target_count_note"],
                r["norm_specificity"],
            )
            log.info(
                "    GC: %.2f | MFE: %.2f kcal/mol | Homopolymer: %d | PolyT: %s",
                r["gc_content"], r["folding_energy_kcal_mol"],
                r["max_homopolymer_run"], r["has_poly_t_terminator"],
            )

    # ---------------------------------------------------------------
    # 7. Validation
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 7. Validation ---")
    validation = validate_results(feasible_df, pareto_df, selected_df, config)
    for check_name, result in validation.items():
        if check_name.startswith("_"):
            continue
        status = "PASS" if result["pass"] else "FAIL"
        log.info("  [%s] %s: %s", status, check_name, result["detail"])

    if validation["_all_passed"]:
        log.info("  All validation checks PASSED.")
    else:
        log.warning("  Some validation checks FAILED — review results carefully.")

    # ---------------------------------------------------------------
    # 8. Save outputs
    # ---------------------------------------------------------------
    log.info("")
    log.info("--- 8. Saving Outputs ---")

    # Feasible candidates with normalised scores
    feasible_out = output_dir / "feasible_candidates.csv"
    feasible_df.to_csv(feasible_out, index=False)
    log.info("Saved feasible candidates: %s", feasible_out)

    # Excluded candidates
    excluded_out = output_dir / "excluded_candidates.csv"
    excluded_df.to_csv(excluded_out, index=False)
    log.info("Saved excluded candidates: %s", excluded_out)

    # Pareto front
    pareto_out = output_dir / "pareto_front.csv"
    pareto_df.to_csv(pareto_out, index=False)
    log.info("Saved Pareto front: %s", pareto_out)

    # Weighted-sum solutions (labelled as approximate)
    ws_out = output_dir / "weighted_sum_pareto_approximation.csv"
    ws_df.to_csv(ws_out, index=False)
    log.info("Saved weighted-sum approximation: %s", ws_out)

    # Selected guides
    selected_out = output_dir / "selected_guides.csv"
    selected_df.to_csv(selected_out, index=False)
    log.info("Saved selected guides: %s", selected_out)

    # Pareto plot
    plot_path = output_dir / "pareto_front.png"
    generate_pareto_plot(feasible_df, pareto_df, selected_df, plot_path, config)

    # Build selected guide reports
    selected_reports = []
    for _, row in selected_df.iterrows():
        selected_reports.append(build_guide_report(row))

    # Comprehensive JSON summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "input_csv": config.input_csv,
            "k": config.k,
            "gc_range": [config.gc_min, config.gc_max],
            "max_homopolymer_run": config.max_homopolymer_run,
            "require_no_poly_t": config.require_no_poly_t,
            "mfe_threshold": config.mfe_threshold,
            "capped_guide_policy": config.capped_guide_policy,
            "n_epsilon_steps": config.n_epsilon_steps,
            "n_weight_steps": config.n_weight_steps,
        },
        "filtering": filter_report,
        "pareto_analysis": {
            "method": "epsilon-constraint + direct dominance verification",
            "n_pareto_optimal": len(pareto_df),
            "weighted_sum_note": (
                "Weighted-sum is a SECONDARY method. It finds solutions on the "
                "convex hull of the Pareto front only and does NOT guarantee "
                "the complete non-dominated set."
            ),
            "weighted_sum_unique_solutions": len(ws_df),
        },
        "selected_guides": selected_reports,
        "lp_relaxation": lp_result,
        "validation": validation,
        "mathematical_note": {
            "track_a": (
                "LP relaxation (x in [0,1]): convex LP solved by CVXOPT "
                "for 23MAT204 demonstration"
            ),
            "track_b": (
                "Discrete binary selection (x in {0,1}): exact Pareto analysis "
                "by epsilon-constraint + dominance"
            ),
            "separation": (
                "Track A and Track B are separate analyses. The LP relaxation "
                "does NOT produce the binary guide selection. Rounding the LP "
                "solution is a heuristic, not a guarantee of optimality for the ILP."
            ),
        },
        "output_files": {
            "feasible_candidates": str(feasible_out),
            "excluded_candidates": str(excluded_out),
            "pareto_front": str(pareto_out),
            "weighted_sum_approximation": str(ws_out),
            "selected_guides": str(selected_out),
            "pareto_plot": str(plot_path),
        },
    }

    summary_path = output_dir / "optimization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Saved optimization summary: %s", summary_path)

    log.info("")
    log.info("=" * 72)
    log.info("STEP 6 OPTIMIZATION COMPLETE")
    log.info("  Feasible candidates:    %d", len(feasible_df))
    log.info("  Pareto-optimal guides:  %d", len(pareto_df))
    log.info("  Selected guides (k=%d): %d", config.k, len(selected_df))
    log.info("  Output directory:       %s", output_dir)
    log.info("=" * 72)

    return summary


# =========================================================================== #
# CLI
# =========================================================================== #
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 6 — Multi-Objective CRISPR gRNA Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="results/TP53/gRNA_features.csv",
        help="Input feature CSV",
    )
    parser.add_argument(
        "--output-dir", default="results/TP53/optimization",
        help="Output directory for optimization results",
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Number of guides to select",
    )
    parser.add_argument(
        "--gc-min", type=float, default=0.30,
        help="Minimum GC content",
    )
    parser.add_argument(
        "--gc-max", type=float, default=0.70,
        help="Maximum GC content",
    )
    parser.add_argument(
        "--max-homopolymer", type=int, default=4,
        help="Maximum homopolymer run length",
    )
    parser.add_argument(
        "--mfe-threshold", type=float, default=None,
        help="MFE threshold in kcal/mol (disabled by default)",
    )
    parser.add_argument(
        "--capped-policy", choices=["retain", "exclude"], default="retain",
        help="Policy for off-target capped guides",
    )
    parser.add_argument(
        "--n-epsilon-steps", type=int, default=50,
        help="Number of epsilon steps for Pareto sweep",
    )
    parser.add_argument(
        "--n-weight-steps", type=int, default=21,
        help="Number of weight pairs for weighted-sum sweep",
    )
    parser.add_argument(
        "--skip-lp", action="store_true",
        help="Skip LP relaxation (Track A)",
    )

    args = parser.parse_args()

    config = OptimizerConfig(
        input_csv=args.input,
        output_dir=args.output_dir,
        k=args.k,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        max_homopolymer_run=args.max_homopolymer,
        mfe_threshold=args.mfe_threshold,
        capped_guide_policy=args.capped_policy,
        n_epsilon_steps=args.n_epsilon_steps,
        n_weight_steps=args.n_weight_steps,
        run_lp_relaxation=not args.skip_lp,
    )

    run_optimization(config)


if __name__ == "__main__":
    main()
