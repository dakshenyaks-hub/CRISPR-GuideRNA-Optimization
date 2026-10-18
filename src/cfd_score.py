"""
src/cfd_score.py

Step 9C -- CFD-Based Off-Target Specificity Scoring (Corrected Implementation)

Implements the published Cutting Frequency Determination (CFD) method
(Doench et al., 2016, Nature Biotechnology 34:184-191).

CORRECTION HISTORY (Step 9A audit -> Step 9C fixes):
  Bug A: Mismatch penalty table was a synthetic approximation.
         FIXED: Replaced with exact empirical values from CRISPOR pickle files.
  Bug B: PAM was hardcoded as "AGG" for all off-target sites.
         FIXED: PAM now extracted from 23-mer Bowtie alignment read_seq.
  Bug C: Censored guides stored computed scores instead of NaN.
         FIXED: cfd_off_target_risk and cfd_specificity_score are NaN when censored.

SOURCE PROVENANCE:
  mismatch_score.pkl: maximilianh/crisporWebsite CFD_Scoring/mismatch_score.pkl
    Git blob SHA: 18f9f169d00f377a2704ac51fa57a9571c571bbb
    Repo master SHA: 3aa198f1371bb15733acd98aaa097aad052cdbe3
  pam_scores.pkl: maximilianh/crisporWebsite CFD_Scoring/pam_scores.pkl
    Git blob SHA: 86068fb0ac99de7fff296bf7181c327e83a39a9c
  License: MIT (LICENSE.txt in crisporWebsite repository)
  Reference: Doench JG et al. Nat Biotechnol 34:184-191 (2016)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cfd_score")


# ===========================================================================
# Authoritative CFD Mismatch Penalty Table (Doench et al., 2016 / CRISPOR)
# ===========================================================================
# Key: (mismatch_type, position)
#   mismatch_type: "rX:dY" -- X = guide RNA base (A/C/G/U), Y = DNA base (A/C/G/T)
#   position: 1-based integer, 1 (PAM-distal) to 20 (PAM-proximal / seed)
# Values: mismatch tolerance in [0.0, 1.0]
# Transcribed exactly from mismatch_score.pkl (CRISPOR repo, blob SHA above).

DOENCH_2016_MISMATCH_SCORES: Dict[Tuple[str, int], float] = {
    ("rA:dA", 1): 1.0, ("rA:dA", 2): 0.727273, ("rA:dA", 3): 0.705882,
    ("rA:dA", 4): 0.636364, ("rA:dA", 5): 0.363636, ("rA:dA", 6): 0.714286,
    ("rA:dA", 7): 0.4375, ("rA:dA", 8): 0.428571, ("rA:dA", 9): 0.6,
    ("rA:dA", 10): 0.882353, ("rA:dA", 11): 0.307692, ("rA:dA", 12): 0.333333,
    ("rA:dA", 13): 0.3, ("rA:dA", 14): 0.533333, ("rA:dA", 15): 0.2,
    ("rA:dA", 16): 0.0, ("rA:dA", 17): 0.133333, ("rA:dA", 18): 0.5,
    ("rA:dA", 19): 0.538462, ("rA:dA", 20): 0.6,
    ("rA:dC", 1): 1.0, ("rA:dC", 2): 0.8, ("rA:dC", 3): 0.611111,
    ("rA:dC", 4): 0.625, ("rA:dC", 5): 0.72, ("rA:dC", 6): 0.714286,
    ("rA:dC", 7): 0.705882, ("rA:dC", 8): 0.733333, ("rA:dC", 9): 0.666667,
    ("rA:dC", 10): 0.555556, ("rA:dC", 11): 0.65, ("rA:dC", 12): 0.722222,
    ("rA:dC", 13): 0.652174, ("rA:dC", 14): 0.466667, ("rA:dC", 15): 0.65,
    ("rA:dC", 16): 0.192308, ("rA:dC", 17): 0.176471, ("rA:dC", 18): 0.4,
    ("rA:dC", 19): 0.375, ("rA:dC", 20): 0.764706,
    ("rA:dG", 1): 0.857143, ("rA:dG", 2): 0.785714, ("rA:dG", 3): 0.428571,
    ("rA:dG", 4): 0.352941, ("rA:dG", 5): 0.5, ("rA:dG", 6): 0.454545,
    ("rA:dG", 7): 0.4375, ("rA:dG", 8): 0.428571, ("rA:dG", 9): 0.571429,
    ("rA:dG", 10): 0.333333, ("rA:dG", 11): 0.4, ("rA:dG", 12): 0.263158,
    ("rA:dG", 13): 0.210526, ("rA:dG", 14): 0.214286, ("rA:dG", 15): 0.272727,
    ("rA:dG", 16): 0.0, ("rA:dG", 17): 0.176471, ("rA:dG", 18): 0.190476,
    ("rA:dG", 19): 0.206897, ("rA:dG", 20): 0.227273,
    ("rC:dA", 1): 1.0, ("rC:dA", 2): 0.909091, ("rC:dA", 3): 0.6875,
    ("rC:dA", 4): 0.8, ("rC:dA", 5): 0.636364, ("rC:dA", 6): 0.928571,
    ("rC:dA", 7): 0.8125, ("rC:dA", 8): 0.875, ("rC:dA", 9): 0.875,
    ("rC:dA", 10): 0.941176, ("rC:dA", 11): 0.307692, ("rC:dA", 12): 0.538462,
    ("rC:dA", 13): 0.7, ("rC:dA", 14): 0.733333, ("rC:dA", 15): 0.066667,
    ("rC:dA", 16): 0.307692, ("rC:dA", 17): 0.466667, ("rC:dA", 18): 0.642857,
    ("rC:dA", 19): 0.461538, ("rC:dA", 20): 0.3,
    ("rC:dC", 1): 0.913043, ("rC:dC", 2): 0.695652, ("rC:dC", 3): 0.5,
    ("rC:dC", 4): 0.5, ("rC:dC", 5): 0.6, ("rC:dC", 6): 0.5,
    ("rC:dC", 7): 0.470588, ("rC:dC", 8): 0.642857, ("rC:dC", 9): 0.619048,
    ("rC:dC", 10): 0.388889, ("rC:dC", 11): 0.25, ("rC:dC", 12): 0.444444,
    ("rC:dC", 13): 0.136364, ("rC:dC", 14): 0.0, ("rC:dC", 15): 0.05,
    ("rC:dC", 16): 0.153846, ("rC:dC", 17): 0.058824, ("rC:dC", 18): 0.133333,
    ("rC:dC", 19): 0.125, ("rC:dC", 20): 0.058824,
    ("rC:dT", 1): 1.0, ("rC:dT", 2): 0.727273, ("rC:dT", 3): 0.866667,
    ("rC:dT", 4): 0.842105, ("rC:dT", 5): 0.571429, ("rC:dT", 6): 0.928571,
    ("rC:dT", 7): 0.75, ("rC:dT", 8): 0.65, ("rC:dT", 9): 0.857143,
    ("rC:dT", 10): 0.866667, ("rC:dT", 11): 0.75, ("rC:dT", 12): 0.714286,
    ("rC:dT", 13): 0.384615, ("rC:dT", 14): 0.35, ("rC:dT", 15): 0.222222,
    ("rC:dT", 16): 1.0, ("rC:dT", 17): 0.466667, ("rC:dT", 18): 0.538462,
    ("rC:dT", 19): 0.428571, ("rC:dT", 20): 0.5,
    ("rG:dA", 1): 1.0, ("rG:dA", 2): 0.636364, ("rG:dA", 3): 0.5,
    ("rG:dA", 4): 0.363636, ("rG:dA", 5): 0.3, ("rG:dA", 6): 0.666667,
    ("rG:dA", 7): 0.571429, ("rG:dA", 8): 0.625, ("rG:dA", 9): 0.533333,
    ("rG:dA", 10): 0.8125, ("rG:dA", 11): 0.384615, ("rG:dA", 12): 0.384615,
    ("rG:dA", 13): 0.3, ("rG:dA", 14): 0.266667, ("rG:dA", 15): 0.142857,
    ("rG:dA", 16): 0.0, ("rG:dA", 17): 0.25, ("rG:dA", 18): 0.666667,
    ("rG:dA", 19): 0.666667, ("rG:dA", 20): 0.7,
    ("rG:dG", 1): 0.714286, ("rG:dG", 2): 0.692308, ("rG:dG", 3): 0.384615,
    ("rG:dG", 4): 0.529412, ("rG:dG", 5): 0.785714, ("rG:dG", 6): 0.681818,
    ("rG:dG", 7): 0.6875, ("rG:dG", 8): 0.615385, ("rG:dG", 9): 0.538462,
    ("rG:dG", 10): 0.4, ("rG:dG", 11): 0.428571, ("rG:dG", 12): 0.529412,
    ("rG:dG", 13): 0.421053, ("rG:dG", 14): 0.428571, ("rG:dG", 15): 0.272727,
    ("rG:dG", 16): 0.0, ("rG:dG", 17): 0.235294, ("rG:dG", 18): 0.47619,
    ("rG:dG", 19): 0.448276, ("rG:dG", 20): 0.428571,
    ("rG:dT", 1): 0.9, ("rG:dT", 2): 0.846154, ("rG:dT", 3): 0.75,
    ("rG:dT", 4): 0.9, ("rG:dT", 5): 0.866667, ("rG:dT", 6): 1.0,
    ("rG:dT", 7): 1.0, ("rG:dT", 8): 1.0, ("rG:dT", 9): 0.642857,
    ("rG:dT", 10): 0.933333, ("rG:dT", 11): 1.0, ("rG:dT", 12): 0.933333,
    ("rG:dT", 13): 0.923077, ("rG:dT", 14): 0.75, ("rG:dT", 15): 0.941176,
    ("rG:dT", 16): 1.0, ("rG:dT", 17): 0.933333, ("rG:dT", 18): 0.692308,
    ("rG:dT", 19): 0.714286, ("rG:dT", 20): 0.9375,
    ("rU:dC", 1): 0.956522, ("rU:dC", 2): 0.84, ("rU:dC", 3): 0.5,
    ("rU:dC", 4): 0.625, ("rU:dC", 5): 0.64, ("rU:dC", 6): 0.571429,
    ("rU:dC", 7): 0.588235, ("rU:dC", 8): 0.733333, ("rU:dC", 9): 0.619048,
    ("rU:dC", 10): 0.5, ("rU:dC", 11): 0.4, ("rU:dC", 12): 0.5,
    ("rU:dC", 13): 0.26087, ("rU:dC", 14): 0.0, ("rU:dC", 15): 0.05,
    ("rU:dC", 16): 0.346154, ("rU:dC", 17): 0.117647, ("rU:dC", 18): 0.333333,
    ("rU:dC", 19): 0.25, ("rU:dC", 20): 0.176471,
    ("rU:dG", 1): 0.857143, ("rU:dG", 2): 0.857143, ("rU:dG", 3): 0.428571,
    ("rU:dG", 4): 0.647059, ("rU:dG", 5): 1.0, ("rU:dG", 6): 0.909091,
    ("rU:dG", 7): 0.6875, ("rU:dG", 8): 1.0, ("rU:dG", 9): 0.923077,
    ("rU:dG", 10): 0.533333, ("rU:dG", 11): 0.666667, ("rU:dG", 12): 0.947368,
    ("rU:dG", 13): 0.789474, ("rU:dG", 14): 0.285714, ("rU:dG", 15): 0.272727,
    ("rU:dG", 16): 0.666667, ("rU:dG", 17): 0.705882, ("rU:dG", 18): 0.428571,
    ("rU:dG", 19): 0.275862, ("rU:dG", 20): 0.090909,
    ("rU:dT", 1): 1.0, ("rU:dT", 2): 0.846154, ("rU:dT", 3): 0.714286,
    ("rU:dT", 4): 0.47619, ("rU:dT", 5): 0.5, ("rU:dT", 6): 0.866667,
    ("rU:dT", 7): 0.875, ("rU:dT", 8): 0.8, ("rU:dT", 9): 0.928571,
    ("rU:dT", 10): 0.857143, ("rU:dT", 11): 0.75, ("rU:dT", 12): 0.8,
    ("rU:dT", 13): 0.692308, ("rU:dT", 14): 0.619048, ("rU:dT", 15): 0.578947,
    ("rU:dT", 16): 0.909091, ("rU:dT", 17): 0.533333, ("rU:dT", 18): 0.666667,
    ("rU:dT", 19): 0.285714, ("rU:dT", 20): 0.5625,
}


# ===========================================================================
# Authoritative CFD PAM Penalty Table (Doench et al., 2016 / CRISPOR)
# ===========================================================================
# Key: 2-character string = last 2 nt of the genomic PAM (N[XY]).
# CRISPOR uses only positions 22-23 of the 23-mer protospacer+PAM.
# Example: TGG PAM -> key "GG"; AAG PAM -> key "AG".
# Transcribed from pam_scores.pkl (CRISPOR repo, blob SHA above).

DOENCH_2016_PAM_SCORES: Dict[str, float] = {
    "AA": 0.0,
    "AC": 0.0,
    "AG": 0.259259259,
    "AT": 0.0,
    "CA": 0.0,
    "CC": 0.0,
    "CG": 0.107142857,
    "CT": 0.0,
    "GA": 0.069444444,
    "GC": 0.022222222,
    "GG": 1.0,
    "GT": 0.016129032,
    "TA": 0.0,
    "TC": 0.0,
    "TG": 0.038961039,
    "TT": 0.0,
}

# Keep backward-compatible alias (used by tests that import DEFAULT_PAM_SCORES)
DEFAULT_PAM_SCORES = DOENCH_2016_PAM_SCORES


def pam_key_from_genomic_pam(pam_str: str) -> Optional[str]:
    """
    Convert a raw genomic PAM string to the 2-nt CRISPOR lookup key.

    SpCas9 PAM = 5'-NGG-3' immediately 3' of the protospacer.
    CRISPOR uses only the last 2 nucleotides as the key.
    e.g. "TGG" -> "GG", "AAG" -> "AG"

    Returns:
        2-character uppercase key, or None if unusable.
    """
    if not pam_str or len(pam_str) < 2:
        return None
    key = pam_str.upper()[-2:]
    if not all(c in "ACGT" for c in key):
        return None
    return key


def calc_cfd_site(
    g_seq: str,
    t_seq: str,
    pam: str,
    mismatch_scores: Optional[Dict[Tuple[str, int], float]] = None,
    pam_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate site-level CFD score (CFD_site) for a single potential off-target locus.

    Replicates the CRISPOR reference implementation exactly.

    Args:
        g_seq: 20-nt guide sequence, DNA alphabet (T not U).
        t_seq: 20-nt off-target protospacer, DNA alphabet.
        pam:   2-nt or 3-nt PAM string; only the LAST 2 chars are used.
               Pass None or empty string to get 0.0 (unknown PAM = no contribution).
        mismatch_scores: Optional override; defaults to DOENCH_2016_MISMATCH_SCORES.
        pam_scores:      Optional override; defaults to DOENCH_2016_PAM_SCORES.

    Returns:
        CFD_site in [0.0, 1.0].
    """
    if mismatch_scores is None:
        mismatch_scores = DOENCH_2016_MISMATCH_SCORES
    if pam_scores is None:
        pam_scores = DOENCH_2016_PAM_SCORES

    g = g_seq.upper()
    t = t_seq.upper()

    if len(g) != 20 or len(t) != 20:
        raise ValueError(
            f"Sequences must be 20-nt. Got guide={len(g)}, target={len(t)}"
        )

    # Mismatch product
    cfd = 1.0
    g_rna = g.replace("T", "U")
    for i in range(20):
        rna_base = g_rna[i]
        dna_base = t[i]
        dna_equiv = "T" if rna_base == "U" else rna_base
        if dna_equiv != dna_base:
            pos = i + 1
            mm_key = (f"r{rna_base}:d{dna_base}", pos)
            factor = mismatch_scores.get(mm_key, 0.0)
            cfd *= factor
            if cfd == 0.0:
                break

    # PAM penalty
    if not pam or len(pam) < 2:
        return 0.0
    pam_key = pam.upper()[-2:]
    pam_factor = pam_scores.get(pam_key, 0.0)
    cfd *= pam_factor

    return round(cfd, 6)


def calc_guide_cfd_score(
    site_cfd_scores: List[float],
    off_target_capped: bool = False,
    incomplete_pam_count: int = 0,
) -> Dict[str, Any]:
    """
    Aggregate site-level CFD scores into guide-level specificity metrics.

    CENSORED HANDLING (Step 9C):
        If off_target_capped=True, cfd_off_target_risk and cfd_specificity_score
        are set to NaN. The partial sum is stored in cfd_partial_risk.

    Returns dict with keys:
        cfd_off_target_risk     float | NaN
        cfd_specificity_score   float | NaN
        cfd_partial_risk        float  (always set)
        cfd_scored_site_count   int
        cfd_incomplete_pam      int
        cfd_scoring_complete    bool
        cfd_scoring_status      str
    """
    partial_risk = float(sum(site_cfd_scores))
    n_scored = len(site_cfd_scores)

    if off_target_capped:
        status = "CENSORED_CAP_REACHED"
        complete = False
        r_cfd = float("nan")
        s_cfd = float("nan")
    else:
        status = "COMPLETE"
        complete = True
        r_cfd = partial_risk
        s_cfd = round(100.0 / (100.0 + r_cfd), 6)

    return {
        "cfd_off_target_risk": r_cfd,
        "cfd_specificity_score": s_cfd,
        "cfd_partial_risk": round(partial_risk, 6),
        "cfd_scored_site_count": n_scored,
        "cfd_incomplete_pam": incomplete_pam_count,
        "cfd_scoring_complete": complete,
        "cfd_scoring_status": status,
    }
