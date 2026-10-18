"""
scripts/step10b_optimize.py

Step 10B -- Integrate Validated CFD Scores into Multi-Objective Optimization

PURPOSE
-------
This script integrates the validated Step 10A CFD-rescored dataset
  results/step10a_cfd_rescored/cfd_guide_features.csv
into the existing optimization pipeline as a CFD-aware variant.

Minimum changes required:
  1. Load Step 10A CFD dataset and deduplicate on guide_id.
  2. Audit and document all duplicate rows (known many-to-many artifact).
  3. Write per-gene enriched feature CSVs to a staging directory.
  4. Apply CFD-aware feasibility filtering.
  5. Run both Track A (LP relaxation, x_i in [0,1]) and Track B (discrete
     Pareto, x_i in {0,1}) with CFD specificity as the specificity axis.
  6. Compare results against Step 8 baselines (read-only, never overwritten).
  7. Run a second identical pass for reproducibility verification.
  8. Emit results/step10b_optimization/<gene>/ and reports/.

Mathematical design choices (documented per spec):
  - Activity   : deep_spcas9_score            -> maximize
  - Specificity: cfd_specificity_score         -> maximize
    (cfd_specificity_score = 100/(100+R) is a MONOTONE DECREASING function of
     cfd_off_target_risk. Only ONE of the two is used in the Pareto objectives
     to avoid redundant / conflicting signals. Choice: cfd_specificity_score.)
  - LP Track A : min-max normalised activity + CFD specificity (equal weight)
  - Pareto Track B: epsilon-constraint on CFD specificity burden,
                    primary objective = activity

WHAT IS NOT CHANGED
-------------------
  - src/grna_optimizer.py        (not modified)
  - src/convex_optimization.py   (not modified)
  - results/step10a_cfd_rescored/ (never overwritten)
  - results/step8_benchmarking/   (read-only comparison)
  - results/<gene>/optimization/  (original Step 6/7 outputs preserved)

Author: CRISPR Design Framework -- Step 10B Integration
"""

from __future__ import annotations

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

from src.grna_optimizer import find_pareto_front

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step10b")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CFD_RESCORED_CSV = PROJECT_ROOT / "results" / "step10a_cfd_rescored" / "cfd_guide_features.csv"
STEP8_REPORT     = PROJECT_ROOT / "results" / "step8_benchmarking" / "step8_report.json"
OUTPUT_BASE      = PROJECT_ROOT / "results" / "step10b_optimization"
REPORTS_DIR      = PROJECT_ROOT / "reports"

GENES = ["TP53", "HBB", "EGFR", "BRCA1"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GC_MIN          = 0.30
GC_MAX          = 0.70
MAX_HOMOPOLYMER = 4
REQUIRE_POLY_T  = True   # has_poly_t_terminator must be False
K               = 5      # select up to K guides per gene

# Feasibility constraint audit (Section 7 of spec):
# Legacy: off_target_count <= 20 (from convex_optimization.py OT_MAX=20).
# Origin: hard-coded in Module 5; no documented biological citation.
# Refers to: raw Bowtie count, NOT CFD-scored risk.
# Decision: NOT applied in Step 10B CFD track because:
#   (a) All uncapped guides already have off_target_count < 999.
#   (b) CFD risk captures off-target quality (weighted scores), not just count.
#   (c) The combination would create REDUNDANT filtering with no scientific basis.
# Instead: capped guides (count=999, CFD risk=NaN) are excluded conservatively.
# The legacy threshold impact is reported as an audit statistic.
LEGACY_OT_MAX = 20


# ===========================================================================
# 1. Load and Deduplicate the CFD Dataset
# ===========================================================================

def load_cfd_dataset() -> tuple[pd.DataFrame, dict]:
    """
    Load results/step10a_cfd_rescored/cfd_guide_features.csv.
    Deduplicate on guide_id -- all duplicate rows are fully identical
    (caused by the many-to-many merge artifact documented in Step 10A report).
    Returns (deduped_df, dedup_audit).
    """
    log.info("Loading Step 10A CFD dataset from %s", CFD_RESCORED_CSV)
    raw = pd.read_csv(CFD_RESCORED_CSV)
    n_raw    = len(raw)
    n_unique = raw["guide_id"].nunique()
    n_dup    = raw["guide_id"].duplicated().sum()

    audit: dict = {
        "source": str(CFD_RESCORED_CSV),
        "n_raw_rows": n_raw,
        "n_unique_guide_ids": n_unique,
        "n_duplicate_rows_removed": int(n_dup),
        "dedup_strategy": (
            "drop_duplicates(subset='guide_id', keep='first'). "
            "All duplicate rows verified to be fully identical across all 44 columns "
            "(max unique values per guide_id group = 1 for every column). "
            "Root cause: many-to-many merge on guide_rna in step9c_cfd.py -- "
            "documented in Step 10A report."
        ),
        "per_gene_raw": {},
        "per_gene_dedup": {},
    }

    for gene in GENES:
        sub = raw[raw["gene"] == gene]
        audit["per_gene_raw"][gene] = {
            "rows": len(sub),
            "unique_ids": int(sub["guide_id"].nunique()),
            "dup_rows": int(sub["guide_id"].duplicated().sum()),
        }

    deduped = raw.drop_duplicates(subset="guide_id", keep="first").reset_index(drop=True)

    for gene in GENES:
        sub = deduped[deduped["gene"] == gene]
        audit["per_gene_dedup"][gene] = {
            "rows": len(sub),
            "complete_cfd": int((sub["cfd_scoring_complete"] == True).sum()),
            "censored_cfd": int((sub["cfd_scoring_complete"] == False).sum()),
        }

    log.info(
        "Dedup: %d raw rows -> %d unique guides (%d duplicate rows removed)",
        n_raw, len(deduped), n_dup,
    )
    return deduped, audit


# ===========================================================================
# 2. Build Per-Gene Enriched Feature CSV
# ===========================================================================

def build_gene_staging_csv(
    gene: str,
    cfd_dedup: pd.DataFrame,
    staging_dir: Path,
) -> tuple[Path, dict]:
    """
    LEFT-JOIN CFD columns onto the canonical feature file using guide_id.
    Verifies: no many-to-many expansion, no silent candidate loss.
    """
    feat_path = PROJECT_ROOT / "results" / gene / "gRNA_features.csv"
    feat_df   = pd.read_csv(feat_path)
    gene_cfd  = cfd_dedup[cfd_dedup["gene"] == gene].copy()

    n_feat = len(feat_df)
    n_cfd  = len(gene_cfd)

    # Verify 1-to-1 within gene_cfd post-dedup
    assert gene_cfd["guide_id"].duplicated().sum() == 0, \
        f"[{gene}] Post-dedup CFD still has duplicate guide_ids!"

    # CFD-specific extra columns to join
    cfd_cols = [
        "guide_id",
        "cfd_off_target_risk", "cfd_specificity_score",
        "cfd_partial_risk", "cfd_scored_site_count",
        "cfd_incomplete_pam", "cfd_scoring_complete",
        "cfd_scoring_status", "on_target_excluded",
        "on_target_exclusion_method", "n_perfect_copy_loci",
    ]
    cfd_slim = gene_cfd[[c for c in cfd_cols if c in gene_cfd.columns]]

    merged  = feat_df.merge(cfd_slim, on="guide_id", how="left")
    n_merged = len(merged)
    n_found  = merged["cfd_scoring_status"].notna().sum()
    n_missing = merged["cfd_scoring_status"].isna().sum()

    join_audit = {
        "gene": gene,
        "feature_file_rows": n_feat,
        "cfd_gene_rows_dedup": n_cfd,
        "merged_rows": n_merged,
        "guides_with_cfd_data": int(n_found),
        "guides_missing_cfd_data": int(n_missing),
        "many_to_many_expansion": int(n_merged - n_feat),
        "join_ok": bool(n_merged == n_feat and n_missing == 0),
    }

    if n_merged != n_feat:
        log.error("[%s] JOIN EXPANDED rows: %d -> %d", gene, n_feat, n_merged)
    if n_missing > 0:
        log.warning("[%s] %d guides missing CFD data", gene, n_missing)
    if join_audit["join_ok"]:
        log.info("[%s] JOIN OK: %d -> %d (no expansion, no loss)", gene, n_feat, n_merged)

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"{gene}_cfd_enriched.csv"
    merged.to_csv(out_path, index=False)
    return out_path, join_audit


# ===========================================================================
# 3. CFD-Aware Normalisation
# ===========================================================================

def normalise_cfd_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute min-max normalised objectives within the feasible set (gene-specific).

    Objectives (Section 3 of spec):
      norm_activity        : deep_spcas9_score           -> [0,1], MAXIMIZE
      norm_cfd_specificity : cfd_specificity_score        -> [0,1], MAXIMIZE
                             (censored guides assigned 0.0 conservative floor)
      norm_ot_burden       : off_target_count (legacy)    -> [0,1], lower=better
      norm_specificity     : 1 - norm_ot_burden (legacy)  -> [0,1], MAXIMIZE

    Monotonic redundancy note (per spec Section 3):
      cfd_specificity_score = 100 / (100 + cfd_off_target_risk)
      is a strictly monotone decreasing function of cfd_off_target_risk.
      They carry IDENTICAL ranking information. Only cfd_specificity_score
      is used as the Pareto specificity axis to avoid redundancy.

    Normalization is GENE-SPECIFIC (within feasible set). Scores are NOT
    comparable across genes.
    """
    df = df.copy()

    # Activity
    E = df["deep_spcas9_score"].values.astype(np.float64)
    E_min, E_max = E.min(), E.max()
    E_range = (E_max - E_min) if E_max > E_min else 1.0
    df["norm_activity"] = (E - E_min) / E_range

    # CFD specificity (primary)
    censored = df["cfd_scoring_complete"].fillna(False) == False
    cfd_spec = df["cfd_specificity_score"].copy().fillna(0.0)  # conservative floor
    S_min, S_max = cfd_spec.min(), cfd_spec.max()
    S_range = (S_max - S_min) if S_max > S_min else 1.0
    df["norm_cfd_specificity"] = (cfd_spec - S_min) / S_range
    df["_censored"] = censored.values

    # Legacy raw OT burden (kept for LP compat)
    O = df["off_target_count"].values.astype(np.float64)
    O_min, O_max = O.min(), O.max()
    O_range = (O_max - O_min) if O_max > O_min else 1.0
    df["norm_ot_burden"]   = (O - O_min) / O_range
    df["norm_specificity"] = 1.0 - df["norm_ot_burden"]

    log.info(
        "Normalisation | E[%.4f,%.4f]->[0,1] | CFD_spec[%.4f,%.4f]->[0,1] | OT[%.0f,%.0f]->[0,1]",
        E_min, E_max, S_min, S_max, O_min, O_max,
    )
    return df


# ===========================================================================
# 4. Biological Feasibility Filter (CFD-aware)
# ===========================================================================

def apply_cfd_feasibility(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Feasibility constraints (documented per Section 7 of spec):
      1. GC content in [GC_MIN, GC_MAX]
      2. max_homopolymer_run <= MAX_HOMOPOLYMER
      3. has_poly_t_terminator == False
      4. deep_spcas9_score not NaN
      5. off_target_capped == False
         -> Rationale: capped guides have cfd_scoring_status='CENSORED_CAP_REACHED'
            and cfd_off_target_risk=NaN. Including them requires imputing an
            arbitrary CFD risk value. Exclusion is the scientifically conservative choice.
            This supersedes the legacy OT<=20 threshold for the CFD track.
    """
    n_total = len(df)
    exclude_mask = pd.Series(False, index=df.index)
    reasons: dict[int, list[str]] = {i: [] for i in df.index}

    nan_act = df["deep_spcas9_score"].isna()
    gc_bad  = (df["gc_content"] < GC_MIN) | (df["gc_content"] > GC_MAX)
    hp_bad  = df["max_homopolymer_run"] > MAX_HOMOPOLYMER
    pt_bad  = df["has_poly_t_terminator"] == True if REQUIRE_POLY_T else pd.Series(False, index=df.index)
    cap_bad = df["off_target_capped"] == True

    for idx in df.index[nan_act]: reasons[idx].append("nan_deep_spcas9_score")
    for idx in df.index[gc_bad]:  reasons[idx].append(f"gc_out_of_range[{GC_MIN},{GC_MAX}]")
    for idx in df.index[hp_bad]:  reasons[idx].append(f"homopolymer_run>{MAX_HOMOPOLYMER}")
    for idx in df.index[pt_bad]:  reasons[idx].append("poly_t_terminator")
    for idx in df.index[cap_bad]: reasons[idx].append("off_target_capped=True (CFD=CENSORED,NaN)")

    exclude_mask = pd.Series(
        [bool(reasons[i]) for i in df.index], index=df.index
    )

    feasible_df = df[~exclude_mask].copy().reset_index(drop=True)
    excl_df     = df[exclude_mask].copy()
    excl_df["exclusion_reasons"] = [
        "; ".join(reasons[i]) for i in excl_df.index
    ]
    excl_df = excl_df.reset_index(drop=True)

    n_legacy_feasible = int((df["off_target_count"] <= LEGACY_OT_MAX).sum())

    audit = {
        "total_candidates": n_total,
        "feasible_candidates": len(feasible_df),
        "excluded_candidates": len(excl_df),
        "exclusion_breakdown": {
            "nan_deep_spcas9_score": int(nan_act.sum()),
            "gc_out_of_range": int(gc_bad.sum()),
            "homopolymer_run_exceeded": int(hp_bad.sum()),
            "poly_t_terminator": int(pt_bad.sum()),
            "off_target_capped_excluded": int(cap_bad.sum()),
        },
        "legacy_ot20_threshold_would_retain": n_legacy_feasible,
        "cfd_constraint_rationale": (
            "Capped guides (OT=999) have cfd_scoring_status=CENSORED and "
            "cfd_off_target_risk=NaN. Excluded as scientifically conservative choice. "
            "Legacy raw OT<=20 NOT applied: CFD risk captures off-target quality "
            "for uncapped guides, making the raw count threshold redundant."
        ),
    }

    log.info(
        "[FEASIBILITY] total=%d feasible=%d excluded=%d "
        "(nan=%d gc=%d hp=%d polyt=%d capped=%d) legacy_ot20_would_keep=%d",
        n_total, len(feasible_df), len(excl_df),
        audit["exclusion_breakdown"]["nan_deep_spcas9_score"],
        audit["exclusion_breakdown"]["gc_out_of_range"],
        audit["exclusion_breakdown"]["homopolymer_run_exceeded"],
        audit["exclusion_breakdown"]["poly_t_terminator"],
        audit["exclusion_breakdown"]["off_target_capped_excluded"],
        n_legacy_feasible,
    )
    return feasible_df, excl_df, audit


# ===========================================================================
# 5. CFD-Aware Pareto Analysis (Track B)
# ===========================================================================

def run_cfd_pareto(
    feasible_df: pd.DataFrame,
    n_epsilon_steps: int = 50,
    k: int = K,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Discrete binary Pareto (x_i in {0,1}) via epsilon-constraint method.

    Primary objective  : maximize norm_activity
    Secondary objective: maximize norm_cfd_specificity
    Method: epsilon-constraint sweep + direct dominance verification.

    NOT convex (integer problem). Exact for n<=10000 via pairwise dominance.
    Returns (pareto_df, weighted_sum_df, full_pareto_size).
    """
    activity = feasible_df["norm_activity"].values
    cfd_spec = feasible_df["norm_cfd_specificity"].values

    burden   = 1.0 - cfd_spec
    eps_vals = np.linspace(burden.min(), burden.max(), n_epsilon_steps)

    selected_indices: set[int] = set()
    for eps in eps_vals:
        feas_eps = np.where(burden <= eps + 1e-12)[0]
        if not feas_eps.size:
            continue
        if len(feas_eps) <= k:
            top = feas_eps
        else:
            top = feas_eps[np.argsort(-activity[feas_eps])[:k]]
        selected_indices.update(int(i) for i in top)

    if not selected_indices:
        return pd.DataFrame(), pd.DataFrame(), 0

    candidates = feasible_df.iloc[sorted(selected_indices)].copy()
    is_pareto  = find_pareto_front(
        candidates["norm_activity"].values,
        candidates["norm_cfd_specificity"].values,
    )
    pareto_df = candidates[is_pareto].copy().reset_index(drop=True)

    full_mask      = find_pareto_front(activity, cfd_spec)
    full_pareto_sz = int(full_mask.sum())

    # Merge with full dominance front
    if not pareto_df.empty and "guide_id" in feasible_df.columns:
        all_ids = set(pareto_df["guide_id"]) | set(feasible_df[full_mask]["guide_id"])
        merged  = feasible_df[feasible_df["guide_id"].isin(all_ids)].copy()
        is_p2   = find_pareto_front(
            merged["norm_activity"].values,
            merged["norm_cfd_specificity"].values,
        )
        pareto_df = merged[is_p2].copy().reset_index(drop=True)

    log.info(
        "Pareto (CFD): eps-constraint->%d candidates -> %d Pareto-optimal "
        "| full dominance=%d",
        len(selected_indices), len(pareto_df), full_pareto_sz,
    )

    # Weighted-sum secondary (21 weight pairs)
    ws_idx: set[int] = set()
    for w1 in np.linspace(0.0, 1.0, 21):
        scores = w1 * activity + (1.0 - w1) * cfd_spec
        ws_idx.update(int(i) for i in np.argsort(-scores)[:k])
    ws_df = feasible_df.iloc[sorted(ws_idx)].copy().reset_index(drop=True)

    return pareto_df, ws_df, full_pareto_sz


# ===========================================================================
# 6. LP Track A (CFD-aware continuous relaxation)
# ===========================================================================

def run_cfd_lp(feasible_df: pd.DataFrame, k: int = K) -> dict:
    """
    LP Track A -- CFD-aware continuous relaxation.

    Formulation (minimisation via CVXOPT):
        min  c^T x
        s.t. sum(x) = k         (cardinality equality)
             0 <= x_i <= 1      (box bounds -- LP relaxation)

        c_i = -(0.5 * norm_activity_i + 0.5 * norm_cfd_specificity_i)

    THIS IS THE LP RELAXATION (x_i in [0,1]).
    It is NOT the binary ILP (x_i in {0,1}).
    Fractional x_i values are REPORTED EXPLICITLY, never silently rounded.

    Convexity:
      Objective: linear (both convex and concave).
      Constraints: affine inequalities (convex feasible region).
      -> This IS a convex optimization problem.
      -> It is NOT the binary mixed-integer program.
    """
    try:
        from cvxopt import matrix, spmatrix, solvers
    except ImportError:
        log.warning("CVXOPT not installed -- skipping LP Track A.")
        return {"status": "skipped", "reason": "cvxopt not installed"}

    solvers.options["show_progress"] = False
    n = len(feasible_df)
    if n == 0:
        return {"status": "empty_feasible_set"}

    activity  = feasible_df["norm_activity"].values.astype(np.float64)
    cfd_spec  = feasible_df["norm_cfd_specificity"].values.astype(np.float64)
    w1, w2    = 0.5, 0.5
    c_vec     = -(w1 * activity + w2 * cfd_spec)

    c_cvx = matrix(c_vec, (n, 1), tc="d")
    rows  = list(range(n)) + list(range(n, 2 * n))
    cols  = list(range(n)) + list(range(n))
    vals  = [-1.0] * n + [1.0] * n
    G_cvx = spmatrix(vals, rows, cols, (2 * n, n), tc="d")
    h_arr = np.zeros(2 * n)
    h_arr[n:] = 1.0
    h_cvx = matrix(h_arr, tc="d")
    A_cvx = matrix(np.ones((1, n), dtype=np.float64), tc="d")
    b_cvx = matrix(np.array([float(min(k, n))]), tc="d")

    log.info("LP Track A (CFD): n=%d, k=%d ...", n, min(k, n))
    sol = solvers.lp(c_cvx, G_cvx, h_cvx, A_cvx, b_cvx)

    if sol["status"] != "optimal":
        return {"status": sol["status"]}

    x_opt = np.array(sol["x"]).flatten()
    obj   = float(sol["primal objective"])
    mu    = float(np.array(sol["y"]).flatten()[0])
    z     = np.array(sol["z"]).flatten()
    alpha, beta = z[:n], z[n:]

    tol        = 1e-6
    fully_sel  = int(np.sum(x_opt > 1.0 - tol))
    fully_rej  = int(np.sum(x_opt < tol))
    fractional = int(np.sum((x_opt > tol) & (x_opt < 1.0 - tol)))
    stat_err   = float(np.max(np.abs(c_vec + mu - alpha + beta)))
    cs_a       = float(np.max(np.abs(alpha * x_opt)))
    cs_b       = float(np.max(np.abs(beta * (1.0 - x_opt))))

    if fractional > 0:
        log.warning(
            "LP Track A: %d FRACTIONAL x_i values in (0,1). "
            "Reported as-is (NOT rounded).", fractional
        )

    top_idx = np.argsort(-x_opt)[:min(10, n)]
    top_vars = [
        {
            "guide_id": str(feasible_df.iloc[i]["guide_id"]),
            "x_value": round(float(x_opt[i]), 6),
            "norm_activity": round(float(activity[i]), 6),
            "norm_cfd_specificity": round(float(cfd_spec[i]), 6),
        }
        for i in top_idx
    ]

    log.info(
        "LP Track A: status=%s obj=%.6f fully_sel=%d frac=%d rej=%d",
        sol["status"], -obj, fully_sel, fractional, fully_rej,
    )
    return {
        "status": sol["status"],
        "objective_value_maximised": round(-obj, 6),
        "weights_used": {"w1_activity": w1, "w2_cfd_specificity": w2},
        "n_variables": n,
        "k": min(k, n),
        "note": (
            "LP RELAXATION x_i in [0,1]. NOT the binary MIP x_i in {0,1}. "
            "Fractional solutions reported exactly, never rounded."
        ),
        "solution_summary": {
            "fully_selected_x_approx_1": fully_sel,
            "fractional_0_lt_x_lt_1": fractional,
            "fully_rejected_x_approx_0": fully_rej,
        },
        "kkt_analysis": {
            "mu_cardinality_dual": round(mu, 6),
            "stationarity_max_residual": round(stat_err, 10),
            "cs_alpha_max": round(cs_a, 10),
            "cs_beta_max": round(cs_b, 10),
            "kkt_satisfied": bool(stat_err < 1e-5 and cs_a < 1e-5 and cs_b < 1e-5),
        },
        "top_lp_variables": top_vars,
    }


# ===========================================================================
# 7. Select Balanced Guides
# ===========================================================================

def select_balanced(pareto_df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    if pareto_df.empty:
        return pd.DataFrame()
    s = 0.5 * pareto_df["norm_activity"].values + 0.5 * pareto_df["norm_cfd_specificity"].values
    p = pareto_df.copy()
    p["balanced_score"] = s
    return p.nlargest(k, "balanced_score").reset_index(drop=True)


# ===========================================================================
# 8. Per-Gene Optimization Runner
# ===========================================================================

def run_gene_optimization(
    gene: str,
    cfd_dedup: pd.DataFrame,
    output_base: Path,
    run_label: str = "run1",
) -> dict:
    log.info("")
    log.info("=" * 72)
    log.info("Step 10B -- Gene: %s  (%s)", gene, run_label)
    log.info("=" * 72)

    staging_dir  = output_base / "staging"
    staging_csv, join_audit = build_gene_staging_csv(gene, cfd_dedup, staging_dir)

    df = pd.read_csv(staging_csv)
    feasible_df, excl_df, feas_audit = apply_cfd_feasibility(df)

    if feasible_df.empty:
        log.error("[%s] EMPTY FEASIBLE SET.", gene)
        return {"gene": gene, "run_label": run_label,
                "error": "empty_feasible_set",
                "join_audit": join_audit,
                "feasibility_audit": feas_audit}

    feasible_df = normalise_cfd_objectives(feasible_df)
    pareto_df, ws_df, full_sz = run_cfd_pareto(feasible_df, k=K)
    lp_result   = run_cfd_lp(feasible_df, k=K)
    selected_df = select_balanced(pareto_df, k=K)

    gene_out = output_base / gene / run_label
    gene_out.mkdir(parents=True, exist_ok=True)

    feasible_df.to_csv(gene_out / "feasible_candidates.csv", index=False)
    excl_df.to_csv(gene_out / "excluded_candidates.csv", index=False)
    pareto_df.to_csv(gene_out / "pareto_front.csv", index=False)
    ws_df.to_csv(gene_out / "weighted_sum_approximation.csv", index=False)
    selected_df.to_csv(gene_out / "selected_guides.csv", index=False)

    # Pareto plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(feasible_df["norm_cfd_specificity"], feasible_df["norm_activity"],
                   c="#d0d0d0", s=8, alpha=0.4, label="Feasible", zorder=1)
        if not pareto_df.empty:
            ax.scatter(pareto_df["norm_cfd_specificity"], pareto_df["norm_activity"],
                       c="#2196F3", s=40, alpha=0.85, label="Pareto-optimal (CFD)",
                       zorder=3, edgecolors="navy", linewidths=0.5)
            pf = pareto_df.sort_values("norm_cfd_specificity")
            ax.plot(pf["norm_cfd_specificity"], pf["norm_activity"],
                    c="#1565C0", linewidth=1.2, alpha=0.6, zorder=2)
        if not selected_df.empty:
            ax.scatter(selected_df["norm_cfd_specificity"], selected_df["norm_activity"],
                       c="#FF1744", s=200, marker="*",
                       label=f"Selected (k={len(selected_df)})",
                       zorder=5, edgecolors="darkred", linewidths=0.8)
        ax.set_xlabel("Normalised CFD Specificity ->", fontsize=11)
        ax.set_ylabel("Normalised On-Target Activity (DeepSpCas9) ->", fontsize=11)
        ax.set_title(
            f"CFD-Aware Multi-Objective Optimisation -- {gene}\n"
            "(Step 10B: epsilon-constraint Pareto, CFD specificity axis)", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(gene_out / "pareto_front.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("[%s] Pareto plot saved.", gene)
    except ImportError:
        log.warning("[%s] matplotlib not available.", gene)

    # Helpers
    def _rng(s: pd.Series) -> dict:
        s2 = s.dropna()
        if s2.empty:
            return {"min": None, "max": None, "mean": None, "std": None}
        return {"min": round(float(s2.min()), 6), "max": round(float(s2.max()), 6),
                "mean": round(float(s2.mean()), 6), "std": round(float(s2.std()), 6)}

    selected_details = []
    for _, row in selected_df.iterrows():
        entry = {
            "guide_id": str(row["guide_id"]),
            "guide_rna": str(row["guide_rna"]),
            "deep_spcas9_score": round(float(row["deep_spcas9_score"]), 6),
            "norm_activity": round(float(row["norm_activity"]), 6),
            "cfd_off_target_risk": (
                round(float(row["cfd_off_target_risk"]), 6)
                if pd.notna(row.get("cfd_off_target_risk")) else None),
            "cfd_specificity_score": (
                round(float(row["cfd_specificity_score"]), 6)
                if pd.notna(row.get("cfd_specificity_score")) else None),
            "norm_cfd_specificity": round(float(row["norm_cfd_specificity"]), 6),
            "balanced_score": round(float(row["balanced_score"]), 6),
            "gc_content": round(float(row["gc_content"]), 4),
            "off_target_count": int(row["off_target_count"]),
            "cfd_scoring_status": str(row.get("cfd_scoring_status", "N/A")),
        }
        selected_details.append(entry)
        log.info(
            "[%s] SELECTED: %s | act=%.4f | cfd_risk=%s | cfd_spec=%s | balanced=%.4f",
            gene, entry["guide_id"], entry["norm_activity"],
            f"{entry['cfd_off_target_risk']:.4f}" if entry["cfd_off_target_risk"] else "NaN",
            f"{entry['cfd_specificity_score']:.4f}" if entry["cfd_specificity_score"] else "NaN",
            entry["balanced_score"],
        )

    summary = {
        "gene": gene, "run_label": run_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "join_audit": join_audit,
        "feasibility_audit": feas_audit,
        "pareto_analysis": {
            "method": "epsilon-constraint (CFD specificity axis) + direct dominance",
            "full_pareto_front_size": full_sz,
            "verified_pareto_optimal_count": len(pareto_df),
            "weighted_sum_unique_solutions": len(ws_df),
        },
        "objective_ranges_feasible": {
            "activity_deep_spcas9": _rng(feasible_df["deep_spcas9_score"]),
            "norm_activity": _rng(feasible_df["norm_activity"]),
            "cfd_off_target_risk": _rng(feasible_df["cfd_off_target_risk"]),
            "cfd_specificity_score": _rng(feasible_df["cfd_specificity_score"]),
            "norm_cfd_specificity": _rng(feasible_df["norm_cfd_specificity"]),
        },
        "selected_guides": selected_details,
        "n_selected": len(selected_details),
        "lp_track_a": lp_result,
        "output_files": {k: str(v) for k, v in {
            "feasible_candidates": gene_out / "feasible_candidates.csv",
            "excluded_candidates": gene_out / "excluded_candidates.csv",
            "pareto_front": gene_out / "pareto_front.csv",
            "weighted_sum": gene_out / "weighted_sum_approximation.csv",
            "selected_guides": gene_out / "selected_guides.csv",
            "pareto_plot": gene_out / "pareto_front.png",
        }.items()},
    }

    with open(gene_out / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ===========================================================================
# 9. Baseline Comparison (read-only)
# ===========================================================================

def compare_baselines(run1: dict[str, dict]) -> dict:
    comp: dict = {
        "step8_source": str(STEP8_REPORT),
        "comparability_note": (
            "Step 8 specificity = 1 - min_max(raw off_target_count). "
            "Step 10B specificity = min_max(cfd_specificity_score). "
            "Different axes, different normalization bases. "
            "Quantitative score comparison is NOT valid. "
            "Qualitative guide-selection comparison IS valid."
        ),
        "per_gene": {},
    }

    if not STEP8_REPORT.exists():
        comp["error"] = "Step 8 report not found -- skipped."
        return comp

    with open(STEP8_REPORT, encoding="utf-8") as f:
        s8 = json.load(f)

    for gene in GENES:
        gs8  = s8.get("per_gene_results", {}).get(gene, {})
        g10b = run1.get(gene, {})

        s8_pareto_id = None
        for method, info in gs8.get("selection_results", {}).items():
            if "pareto" in method.lower():
                ids = info.get("selected_guide_ids", [])
                if ids:
                    s8_pareto_id = ids[0]
                    break

        s10b_ids = [g["guide_id"] for g in g10b.get("selected_guides", [])]

        comp["per_gene"][gene] = {
            "step8_pareto_selected": s8_pareto_id,
            "step10b_cfd_selected": s10b_ids,
            "same_top_guide": s8_pareto_id in s10b_ids if s8_pareto_id else None,
            "step8_feasible_count": gs8.get("feasible_count", "N/A"),
            "step10b_feasible_count": g10b.get("feasibility_audit", {}).get("feasible_candidates", "N/A"),
            "step8_pareto_size": gs8.get("pareto_size", "N/A"),
            "step10b_pareto_size": g10b.get("pareto_analysis", {}).get("verified_pareto_optimal_count", "N/A"),
            "direct_score_comparison_valid": False,
        }
    return comp


# ===========================================================================
# 10. Reproducibility Verification
# ===========================================================================

def verify_reproducibility(r1: dict[str, dict], r2: dict[str, dict]) -> dict:
    repro: dict = {"all_passed": True, "per_gene": {}}

    for gene in GENES:
        g1, g2 = r1.get(gene, {}), r2.get(gene, {})
        if "error" in g1 or "error" in g2:
            repro["per_gene"][gene] = {"pass": False, "reason": "error in one run"}
            repro["all_passed"] = False
            continue

        f1 = g1.get("feasibility_audit", {}).get("feasible_candidates", -1)
        f2 = g2.get("feasibility_audit", {}).get("feasible_candidates", -2)
        p1 = g1.get("pareto_analysis", {}).get("verified_pareto_optimal_count", -1)
        p2 = g2.get("pareto_analysis", {}).get("verified_pareto_optimal_count", -2)
        i1 = sorted(g["guide_id"] for g in g1.get("selected_guides", []))
        i2 = sorted(g["guide_id"] for g in g2.get("selected_guides", []))
        o1 = sorted(round(g["balanced_score"], 5) for g in g1.get("selected_guides", []))
        o2 = sorted(round(g["balanced_score"], 5) for g in g2.get("selected_guides", []))

        passed = (f1 == f2) and (p1 == p2) and (i1 == i2) and (o1 == o2)
        if not passed:
            repro["all_passed"] = False

        repro["per_gene"][gene] = {
            "pass": passed,
            "feasible_count": {"run1": f1, "run2": f2, "match": f1 == f2},
            "pareto_size": {"run1": p1, "run2": p2, "match": p1 == p2},
            "selected_ids": {"run1": i1, "run2": i2, "match": i1 == i2},
            "balanced_scores": {"run1": o1, "run2": o2, "match": o1 == o2},
        }
    return repro


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    log.info("=" * 72)
    log.info("STEP 10B -- CFD-AWARE MULTI-OBJECTIVE OPTIMIZATION")
    log.info("=" * 72)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cfd_dedup, dedup_audit = load_cfd_dataset()

    log.info("--- RUN 1 ---")
    run1: dict[str, dict] = {}
    for gene in GENES:
        run1[gene] = run_gene_optimization(gene, cfd_dedup, OUTPUT_BASE, "run1")

    log.info("--- RUN 2 (reproducibility) ---")
    run2: dict[str, dict] = {}
    for gene in GENES:
        run2[gene] = run_gene_optimization(gene, cfd_dedup, OUTPUT_BASE, "run2")

    baseline_cmp = compare_baselines(run1)
    repro        = verify_reproducibility(run1, run2)

    final = {
        "task": "Step 10B -- Integrate Validated CFD Scores into Multi-Objective Optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cfd_dataset": {"source": str(CFD_RESCORED_CSV), "deduplication_audit": dedup_audit},
        "configuration": {
            "genes": GENES, "k_guides_to_select": K,
            "gc_range": [GC_MIN, GC_MAX],
            "max_homopolymer_run": MAX_HOMOPOLYMER,
            "require_no_poly_t": REQUIRE_POLY_T,
            "capped_guide_policy": "exclude (CFD=NaN, scientifically conservative)",
            "legacy_ot20_threshold_applied": False,
            "legacy_ot20_rationale": (
                "Not applied: origin undocumented (hard-coded in Module 5). "
                "Refers to raw OT count, not CFD risk. "
                "CFD risk captures quality of uncapped off-targets; "
                "raw count threshold is redundant and creates unsupported double-filtering."
            ),
        },
        "objective_definitions": {
            "activity": {
                "source": "deep_spcas9_score", "direction": "maximize",
                "normalization": "min-max within gene feasible set",
                "normalized_column": "norm_activity",
            },
            "specificity": {
                "source": "cfd_specificity_score = 100/(100+cfd_off_target_risk)",
                "direction": "maximize",
                "normalization": "min-max within gene feasible set; censored=0.0 floor",
                "normalized_column": "norm_cfd_specificity",
                "monotone_redundancy_note": (
                    "cfd_specificity_score is a strictly monotone decreasing function "
                    "of cfd_off_target_risk. Only ONE is used in Pareto to avoid redundancy. "
                    "cfd_off_target_risk is NOT simultaneously optimized."
                ),
            },
        },
        "convexity_assessment": {
            "track_a_lp": (
                "LP relaxation (x_i in [0,1]): linear objective, affine constraints. "
                "IS a convex optimization problem. "
                "Is NOT the binary MIP (x_i in {0,1}). "
                "Fractional x_i values reported exactly, never rounded."
            ),
            "track_b_pareto": (
                "Discrete binary Pareto (x_i in {0,1}): epsilon-constraint + dominance. "
                "Integer problem -- NOT convex in general. "
                "Results are exact via pairwise dominance for n<=10000."
            ),
        },
        "per_gene_results": run1,
        "baseline_comparison": baseline_cmp,
        "reproducibility": repro,
        "limitations": [
            "CFD specificity normalization is gene-specific. Scores NOT comparable across genes.",
            "30062 censored guides excluded (OT=999, CFD=NaN). True Pareto may shift if scored.",
            "DeepSpCas9 is a computational prediction, NOT experimental validation.",
            "CFD scoring is a computational model, NOT experimental off-target data.",
            "No wet-lab validation data exists in this project.",
            "LP Track A may return fractional x_i; these are exact for the LP relaxation "
            "but do not directly correspond to a binary guide selection.",
            "Epsilon-constraint uses n=50 steps; more steps may reveal additional Pareto guides.",
        ],
        "recommended_next_step": (
            "Review per-gene Pareto fronts. If trade-off curves are well-defined "
            "and reproducibility passes, proceed to final research analysis. "
            "Consider a CFD risk imputation experiment (assign max-observed risk "
            "to censored guides) to assess sensitivity to the large censored fraction."
        ),
    }

    repro_ok  = repro.get("all_passed", False)
    n_errors  = sum(1 for r in run1.values() if "error" in r)
    join_ok   = all(
        run1[g].get("join_audit", {}).get("join_ok", False)
        for g in GENES if "error" not in run1.get(g, {})
    )

    if n_errors == 0 and repro_ok and join_ok:
        status = "READY FOR FINAL RESEARCH ANALYSIS"
    elif n_errors > 0 or not join_ok:
        status = "NOT READY FOR FINAL RESEARCH ANALYSIS"
    else:
        status = "READY AFTER SPECIFIC FIXES"

    final["final_status"] = status
    log.info("FINAL STATUS: %s", status)

    for path in [REPORTS_DIR / "step10b_optimization_summary.json",
                 OUTPUT_BASE / "step10b_optimization_summary.json"]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, default=str)
        log.info("Saved: %s", path)

    log.info("Step 10B complete. Outputs: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()

