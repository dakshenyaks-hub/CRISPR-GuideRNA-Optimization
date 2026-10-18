"""
src/convex_optimization.py

Module 5 — Convex (Linear-Program) Optimization Stage
Multi-Objective CRISPR Guide RNA Design Framework
======================================================

PURPOSE
-------
Given the per-guide feature scores produced by Module 3 (feature_extraction.py)
and the scoring weights of Module 4 (optimisation.py), this module selects the
best K=10 guide RNAs via a Linear Program.  Biological feasibility constraints
are applied BEFORE the solver, reducing the candidate set to guides that satisfy
experimentally motivated bounds.

INPUT  : data/gRNA_features_specificity.csv   (2 860 candidate gRNAs)
OUTPUT : data/convex_optimized_guides.csv

Do NOT modify optimisation.py or any other existing module.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATHEMATICAL FORMULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Notation
--------
  n   = total number of guides (2 860)
  i   = guide index  (0 … n-1)

Per-guide scores (computed inside this module, same formula as optimisation.py)
-------------------------------------------------------------------------------
  E_i  = efficiency_score_i            ∈ [0, 1]   (min-max normalised Doench proxy)
  S_i  = specificity_score_i           ∈ [0, 1]   (70 % burden + 30 % divergence)
  R_i  = structural_suitability_score_i ∈ [0, 1]  (Tm desirability + dG + MFE)

Objective coefficient (guide utility)
--------------------------------------
  c_i = 0.35·E_i + 0.40·S_i + 0.25·R_i    ∈ [0, 1]

Decision variable
-----------------
  x_i ∈ [0, 1]     continuous selection level of guide i
                    (LP relaxation; since objective coefficients > 0 and
                     constraints are box + budget, the LP optimum is attained
                     at binary vertices — see note below)

Biological feasibility indicator
---------------------------------
  u_i = 1  if ALL of the following hold:
              0.40 ≤ GC_i     ≤ 0.60      (GC content in productive range)
              55.0 ≤ Tm_i     ≤ 65.0      (melting temperature, °C)
              off_target_count_i ≤ 20     (low off-target burden)
              off_target_capped_i = False  (count not truncated by -k 1000)
        = 0  otherwise (guide is excluded from the feasible set)

Optimization problem (primal form)
------------------------------------
  maximize    Σᵢ  cᵢ · xᵢ
  subject to  0  ≤  xᵢ  ≤  uᵢ    ∀ i   (box constraint; capped if uᵢ=0)
              Σᵢ  xᵢ  ≤  K               (budget constraint;  K = 10)

  If fewer than K guides satisfy all biological constraints, the budget is
  automatically non-binding and every feasible guide is selected at xᵢ = 1.

Why is this a Linear Program?
-------------------------------
  • The objective   Σ cᵢ xᵢ   is linear in x (inner product of fixed cᵢ and xᵢ).
  • Every constraint (box bounds, budget) is a linear (affine) inequality in x.
  A problem with a linear objective and affine constraints is, by definition, a
  Linear Program — a special case of convex optimization.

Why is the feasible set convex?
---------------------------------
  Each constraint is a halfspace (an affine inequality), and halfspaces are
  convex sets.  The intersection of any collection of convex sets is convex.
  Therefore the feasible region { x : 0 ≤ xᵢ ≤ uᵢ, Σxᵢ ≤ K } is convex.

Why is the problem convex?
-----------------------------
  An optimization problem is convex when:
    (i)  the objective is convex (linear functions are both convex and concave),
    (ii) the constraints define a convex feasible region (shown above).
  Both conditions hold, so this is a convex optimization problem.

Note on LP vertex property
---------------------------
  For a linear program with a bounded polytope feasible set (as here), the
  optimum is always attained at a vertex of the polytope.  With box constraints
  [0, uᵢ] and a budget Σxᵢ ≤ K, the vertices are binary (0/1) selections.
  The LP relaxation therefore gives the same solution as the Integer Program
  when the number of feasible guides ≤ K (all xᵢ → 1) or when the top-K
  guides by cᵢ can be selected within budget (xᵢ ∈ {0,1} at optimum).

CVXOPT standard (minimization) form
--------------------------------------
  CVXOPT solves:
      minimize    cᵀ x
      subject to  G x ≤ h

  Conversion: maximising Σ cᵢ xᵢ  ≡  minimising Σ (−cᵢ) xᵢ
              → pass objective vector  q = −c  to CVXOPT.

  Constraint encoding (all written as ≤):
      −xᵢ  ≤  0              (lower bound  xᵢ ≥ 0)
       xᵢ  ≤  uᵢ             (upper bound, 0 or 1 depending on feasibility)
      Σ xᵢ ≤  K              (budget)

  G is a (2n+1) × n matrix; h is a (2n+1) vector.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Author  : CRISPR Design Framework — Bioinformatics Development Team
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CVXOPT import with a clear error message if missing
# ---------------------------------------------------------------------------
try:
    import cvxopt
    from cvxopt import matrix, solvers
except ImportError:  # pragma: no cover
    sys.exit(
        "ERROR: cvxopt is not installed.\n"
        "Install it with:  pip install cvxopt\n"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convex_optimization")

# ---------------------------------------------------------------------------
# Hard-coded constants — keep in one place for clarity
# ---------------------------------------------------------------------------
K: int = 10                   # maximum guides to select (budget)

# Objective weights (must match optimisation.py Module 4)
W_EFFICIENCY:   float = 0.35
W_SPECIFICITY:  float = 0.40
W_STRUCTURAL:   float = 0.25

# Specificity sub-weights (must match optimisation.py)
W_BURDEN:       float = 0.70
W_DIVERGENCE:   float = 0.30

# Biological feasibility bounds
GC_LOW:         float = 0.40
GC_HIGH:        float = 0.60
TM_LOW:         float = 55.0
TM_HIGH:        float = 65.0
OT_MAX:         int   = 20    # max off-target count (strictly; capped guides fail)

# Tm desirability window (matches optimisation.py defaults)
TM_TOLERANCE:   float = 5.0


# ===========================================================================
# Score computation (mirrors optimisation.py — no import to avoid coupling)
# ===========================================================================

def _min_max_normalize(s: pd.Series, maximize: bool) -> pd.Series:
    """Scale series to [0,1]. Constant series → 0.5 (avoid /0)."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    scaled = (s - lo) / (hi - lo)
    return scaled if maximize else 1.0 - scaled


def _target_range_desirability(
    s: pd.Series, low: float, high: float, tolerance: float
) -> pd.Series:
    """Derringer-Suich plateau desirability for Tm."""
    def _score(x: float) -> float:
        if low <= x <= high:
            return 1.0
        if low - tolerance <= x < low:
            return (x - (low - tolerance)) / tolerance
        if high < x <= high + tolerance:
            return (high + tolerance - x) / tolerance
        return 0.0
    return s.apply(_score)


def _impute_mismatch(df: pd.DataFrame) -> pd.Series:
    """NaN mismatch_score (zero off-targets) → dataset max (best case)."""
    filled = df["mismatch_score"].copy()
    best = filled.max() if filled.notna().any() else 1.0
    return filled.fillna(best)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute efficiency_score, specificity_score, structural_suitability_score,
    and objective_coefficient c_i for every guide.
    Mirrors the formulas in optimisation.py compute_objective_scores().
    """
    out = pd.DataFrame(index=df.index)

    # -- Efficiency --
    out["efficiency_score"] = _min_max_normalize(
        df["doench2016_proxy_score"], maximize=True
    )

    # -- Specificity: 70 % burden + 30 % divergence --
    raw_counts = df["off_target_count"].copy()
    if "off_target_capped" in df.columns:
        dataset_max = raw_counts.max()
        raw_counts = raw_counts.where(
            ~df["off_target_capped"].astype(bool), other=dataset_max
        )
    burden_norm = _min_max_normalize(raw_counts, maximize=False)
    if "off_target_capped" in df.columns:
        burden_norm = burden_norm.where(
            ~df["off_target_capped"].astype(bool), other=0.0
        )
    divergence_norm = _min_max_normalize(_impute_mismatch(df), maximize=True)
    out["specificity_score"] = W_BURDEN * burden_norm + W_DIVERGENCE * divergence_norm

    # -- Structural suitability: Tm + dG + MFE --
    tm_norm  = _target_range_desirability(
        df["melting_temp_c"], TM_LOW, TM_HIGH, TM_TOLERANCE
    )
    dg_norm  = _min_max_normalize(df["delta_g_kcal_mol"],         maximize=False)
    mfe_norm = _min_max_normalize(df["folding_energy_kcal_mol"],  maximize=True)
    out["structural_suitability_score"] = (tm_norm + dg_norm + mfe_norm) / 3.0

    # -- Objective coefficient c_i --
    out["objective_coefficient"] = (
        W_EFFICIENCY * out["efficiency_score"]
        + W_SPECIFICITY * out["specificity_score"]
        + W_STRUCTURAL  * out["structural_suitability_score"]
    )

    return out


# ===========================================================================
# Biological feasibility filter
# ===========================================================================

def apply_biological_constraints(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series: True iff guide i satisfies ALL constraints:
        GC_LOW  ≤ gc_content      ≤ GC_HIGH
        TM_LOW  ≤ melting_temp_c  ≤ TM_HIGH
        off_target_count          ≤ OT_MAX
        off_target_capped         == False    (count is exact, not a lower bound)

    Constraint parameters logged so they appear in the run record.
    """
    log.info(
        "Biological constraints: GC ∈ [%.2f, %.2f] | Tm ∈ [%.1f, %.1f] °C | "
        "off_target_count ≤ %d | off_target_capped = False",
        GC_LOW, GC_HIGH, TM_LOW, TM_HIGH, OT_MAX,
    )

    gc_ok = df["gc_content"].between(GC_LOW, GC_HIGH)
    tm_ok = df["melting_temp_c"].between(TM_LOW, TM_HIGH)
    ot_ok = df["off_target_count"] <= OT_MAX

    capped_ok = pd.Series(True, index=df.index)
    if "off_target_capped" in df.columns:
        capped_ok = ~df["off_target_capped"].astype(bool)

    feasible = gc_ok & tm_ok & ot_ok & capped_ok

    log.info(
        "Constraint breakdown — GC: %d | Tm: %d | OT count ≤ %d: %d | "
        "Not capped: %d | ALL feasible: %d / %d",
        gc_ok.sum(), tm_ok.sum(), OT_MAX, ot_ok.sum(),
        capped_ok.sum(), feasible.sum(), len(df),
    )
    return feasible


# ===========================================================================
# LP solver
# ===========================================================================

def solve_lp(c_vec: np.ndarray, u_vec: np.ndarray) -> np.ndarray:
    """
    Solve the Linear Program:

        minimize    qᵀ x          (q = −c, converting maximization)
        subject to  G x ≤ h

    where G and h encode:
        −xᵢ ≤  0           (lower bound)
         xᵢ ≤  uᵢ          (upper bound; 0 for infeasible guides)
        Σ xᵢ ≤  K           (budget)

    Returns the optimal x vector as a numpy array.
    If the feasible set is empty (all u_i = 0), raises RuntimeError.
    """
    n = len(c_vec)
    if u_vec.sum() == 0:
        raise RuntimeError(
            "Feasible set is EMPTY: no guide satisfies all biological constraints. "
            "Relax one or more constraints before running the LP."
        )

    # Objective: minimise −c (equivalent to maximising c)
    q = matrix(-c_vec, tc="d")          # (n,1) column vector, double

    # G matrix: (2n + 1) rows × n columns
    # Rows 0..n-1   : −I  (lower bounds  −xᵢ ≤ 0)
    # Rows n..2n-1  : +I  (upper bounds   xᵢ ≤ uᵢ)
    # Row  2n       : 1ᵀ  (budget         Σxᵢ ≤ K)
    G_np = np.zeros((2 * n + 1, n), dtype=float)
    G_np[:n,  :] = -np.eye(n)           # lower bounds
    G_np[n:2*n, :] =  np.eye(n)         # upper bounds
    G_np[2*n, :] =  np.ones(n)          # budget row

    h_np = np.zeros(2 * n + 1, dtype=float)
    h_np[:n]    = 0.0                   # lower bound RHS: −xᵢ ≤ 0
    h_np[n:2*n] = u_vec                 # upper bound RHS:  xᵢ ≤ uᵢ
    h_np[2*n]   = float(K)             # budget RHS:       Σxᵢ ≤ K

    # CVXOPT expects column-major Fortran order
    G = matrix(G_np, tc="d")
    h = matrix(h_np, tc="d")

    solvers.options["show_progress"] = False  # suppress CVXOPT iteration output
    sol = solvers.lp(q, G, h)

    if sol["status"] != "optimal":
        raise RuntimeError(
            f"CVXOPT did not find an optimal solution (status='{sol['status']}'). "
            "Check that the feasible set is non-empty and bounded."
        )

    return np.array(sol["x"]).flatten()


# ===========================================================================
# Orchestration
# ===========================================================================

def run_pipeline(
    input_csv: str = "data/gRNA_features_specificity.csv",
    output_csv: str = "data/convex_optimized_guides.csv",
) -> pd.DataFrame:
    """
    End-to-end pipeline:
      1. Load feature CSV.
      2. Compute per-guide objective scores (E, S, R) and coefficient c_i.
      3. Apply biological constraints → feasibility indicator u_i.
      4. Solve LP with CVXOPT.
      5. Select and rank guides by x_i · c_i (objective contribution).
      6. Save and return the output DataFrame.
    """
    log.info("Loading feature table from '%s' ...", input_csv)
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"No guides found in '{input_csv}'")
    log.info("Loaded %d candidate guides.", len(df))

    # ------------------------------------------------------------------
    # Step 1: compute scores
    # ------------------------------------------------------------------
    log.info(
        "Objective weights — efficiency: %.2f | specificity: %.2f | structural: %.2f",
        W_EFFICIENCY, W_SPECIFICITY, W_STRUCTURAL,
    )
    log.info(
        "Specificity sub-weights — off-target burden: %.2f | mismatch divergence: %.2f",
        W_BURDEN, W_DIVERGENCE,
    )
    scores = compute_scores(df)
    df = pd.concat([df, scores], axis=1)

    # ------------------------------------------------------------------
    # Step 2: biological feasibility
    # ------------------------------------------------------------------
    feasible_mask = apply_biological_constraints(df)
    n_feasible = feasible_mask.sum()

    if n_feasible == 0:
        log.error(
            "EMPTY FEASIBLE SET — no guide satisfies all biological constraints. "
            "Relax one or more bounds (GC, Tm, off-target) before re-running."
        )
        sys.exit(1)

    log.info(
        "Budget K = %d | Feasible guides = %d  →  %s",
        K,
        n_feasible,
        f"selecting all {n_feasible} (< K)" if n_feasible <= K
        else f"LP will select best {K}",
    )

    # ------------------------------------------------------------------
    # Step 3: build LP inputs
    # ------------------------------------------------------------------
    c_vec = df["objective_coefficient"].to_numpy(dtype=float)
    u_vec = feasible_mask.to_numpy(dtype=float)   # 1.0 feasible, 0.0 infeasible

    # ------------------------------------------------------------------
    # Step 4: solve
    # ------------------------------------------------------------------
    log.info("Solving LP via CVXOPT ...")
    x_opt = solve_lp(c_vec, u_vec)

    # ------------------------------------------------------------------
    # Step 5: build output
    # ------------------------------------------------------------------
    df["selection_variable"] = x_opt
    df["objective_contribution"] = df["selection_variable"] * df["objective_coefficient"]

    # Keep only selected guides (x_i > numerical tolerance)
    SELECTION_THRESHOLD = 1e-6
    selected = df[df["selection_variable"] > SELECTION_THRESHOLD].copy()
    selected = selected.sort_values("objective_contribution", ascending=False)
    selected["rank"] = range(1, len(selected) + 1)

    log.info(
        "LP solution: %d guide(s) selected | "
        "total objective = %.6f | "
        "mean c_i = %.4f",
        len(selected),
        selected["objective_contribution"].sum(),
        selected["objective_coefficient"].mean(),
    )

    # ------------------------------------------------------------------
    # Step 6: save
    # ------------------------------------------------------------------
    output_columns = [
        "rank",
        "guide_rna",
        "selection_variable",
        "objective_coefficient",
        "objective_contribution",
        "efficiency_score",
        "specificity_score",
        "structural_suitability_score",
        "gc_content",
        "melting_temp_c",
        "off_target_count",
        "off_target_capped",
    ]
    available = [c for c in output_columns if c in selected.columns]
    output_df = selected[available].reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    log.info("Saved %d optimized guides to '%s'", len(output_df), output_csv)

    # ------------------------------------------------------------------
    # Summary table (printed to stdout for quick inspection)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"CONVEX LP OPTIMIZATION RESULTS   (K={K})")
    print("=" * 78)
    print(f"  Total candidates   : {len(df)}")
    print(f"  Feasible guides    : {n_feasible}")
    print(f"  Guides selected    : {len(output_df)}")
    print(f"  Total objective    : {selected['objective_contribution'].sum():.6f}")
    print(f"  Constraints:")
    print(f"    GC content       in [{GC_LOW:.2f}, {GC_HIGH:.2f}]")
    print(f"    Melting temp     in [{TM_LOW:.1f}, {TM_HIGH:.1f}] C")
    print(f"    Off-target count <= {OT_MAX}")
    print(f"    Off-target capped= False  (must be exact count)")
    print(f"  Budget K           : {K}")
    print("=" * 78)
    print(output_df.to_string(index=False))
    print("=" * 78 + "\n")

    return output_df


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convex (LP) optimization of CRISPR gRNA candidates. "
            "Maximizes Σ c_i x_i subject to biological feasibility and budget K."
        )
    )
    parser.add_argument(
        "--input",
        default="data/gRNA_features_specificity.csv",
        help="Feature CSV produced by feature_extraction.py (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data/convex_optimized_guides.csv",
        help="Output CSV for selected guides (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(input_csv=args.input, output_csv=args.output)
