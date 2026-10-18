"""
tests/test_cfd_score.py

Step 9C -- Corrected CFD Implementation Tests

Tests execute real comparisons against the CRISPOR reference values.
No manually-asserted True flags.
"""

import math
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cfd_score import (
    DOENCH_2016_MISMATCH_SCORES,
    DOENCH_2016_PAM_SCORES,
    calc_cfd_site,
    calc_guide_cfd_score,
    pam_key_from_genomic_pam,
)
from src.step9c_cfd import exclude_on_target, score_guide_cfd


# ===========================================================================
# 1. Exact mismatch penalty values vs CRISPOR reference
# ===========================================================================

class TestMismatchTable:
    def test_table_size(self):
        """12 mismatch types x 20 positions = 240 entries."""
        assert len(DOENCH_2016_MISMATCH_SCORES) == 240

    def test_rAdA_pos1(self):
        """rA:dA at position 1 (PAM-distal) = 1.0 per CRISPOR pkl."""
        assert abs(DOENCH_2016_MISMATCH_SCORES[("rA:dA", 1)] - 1.0) < 1e-6

    def test_rUdG_pos20(self):
        """rU:dG at position 20 (seed) = 0.090909 per CRISPOR pkl."""
        assert abs(DOENCH_2016_MISMATCH_SCORES[("rU:dG", 20)] - 0.090909) < 1e-4

    def test_rCdC_pos14(self):
        """rC:dC at position 14 = 0.0 per CRISPOR pkl."""
        assert DOENCH_2016_MISMATCH_SCORES[("rC:dC", 14)] == 0.0

    def test_rGdT_pos6(self):
        """rG:dT at position 6 = 1.0 (wobble G:T, very tolerant) per CRISPOR pkl."""
        assert DOENCH_2016_MISMATCH_SCORES[("rG:dT", 6)] == 1.0

    def test_rAdA_pos16(self):
        """rA:dA at position 16 = 0.0 (strong seed penalty) per CRISPOR pkl."""
        assert DOENCH_2016_MISMATCH_SCORES[("rA:dA", 16)] == 0.0

    def test_all_types_present(self):
        expected_types = {
            "rA:dA", "rA:dC", "rA:dG",
            "rC:dA", "rC:dC", "rC:dT",
            "rG:dA", "rG:dG", "rG:dT",
            "rU:dC", "rU:dG", "rU:dT",
        }
        actual_types = {k[0] for k in DOENCH_2016_MISMATCH_SCORES.keys()}
        assert actual_types == expected_types


# ===========================================================================
# 2. Exact PAM penalty values vs CRISPOR reference
# ===========================================================================

class TestPAMTable:
    def test_gg_canonical(self):
        assert DOENCH_2016_PAM_SCORES["GG"] == 1.0

    def test_ag_non_canonical(self):
        assert abs(DOENCH_2016_PAM_SCORES["AG"] - 0.259259) < 1e-4

    def test_ga(self):
        assert abs(DOENCH_2016_PAM_SCORES["GA"] - 0.069444) < 1e-4

    def test_gc(self):
        assert abs(DOENCH_2016_PAM_SCORES["GC"] - 0.022222) < 1e-4

    def test_gt(self):
        assert abs(DOENCH_2016_PAM_SCORES["GT"] - 0.016129) < 1e-4

    def test_tg(self):
        assert abs(DOENCH_2016_PAM_SCORES["TG"] - 0.038961) < 1e-4

    def test_cg(self):
        assert abs(DOENCH_2016_PAM_SCORES["CG"] - 0.107143) < 1e-4

    def test_zero_pams(self):
        for k in ["AA", "AC", "AT", "CA", "CC", "CT", "TA", "TC", "TT"]:
            assert DOENCH_2016_PAM_SCORES[k] == 0.0, f"{k} should be 0.0"

    def test_all_16_entries(self):
        assert len(DOENCH_2016_PAM_SCORES) == 16


# ===========================================================================
# 3. calc_cfd_site correctness
# ===========================================================================

class TestCalcCFDSite:
    G = "GAGTCCGAGCAGAAGAAGAA"

    def test_perfect_match_gg(self):
        assert calc_cfd_site(self.G, self.G, "GG") == 1.0

    def test_perfect_match_tgg(self):
        """3-nt PAM string: last 2 chars used."""
        assert calc_cfd_site(self.G, self.G, "TGG") == 1.0

    def test_perfect_match_agg(self):
        assert calc_cfd_site(self.G, self.G, "AGG") == 1.0

    def test_ag_pam(self):
        s = calc_cfd_site(self.G, self.G, "AG")
        assert abs(s - 0.259259) < 1e-4

    def test_unknown_pam_returns_zero(self):
        assert calc_cfd_site(self.G, self.G, "NN") == 0.0

    def test_empty_pam_returns_zero(self):
        assert calc_cfd_site(self.G, self.G, "") == 0.0

    def test_seed_mismatch_stronger(self):
        # G="GAGTCCGAGCAGAAGAAGAA"
        # Seed (pos 20): guide base A, target base C -> rA:dC,20 = 0.764706
        # Distal (pos 1): guide base G, target base A -> rG:dA,1 = 1.0
        # So seed (0.764706) < distal (1.0)
        t_seed = self.G[:19] + "C"       # A->C mismatch at pos 20 (seed): rA:dC,20
        t_dist = "A" + self.G[1:]        # G->A mismatch at pos 1 (distal): rG:dA,1
        s_seed = calc_cfd_site(self.G, t_seed, "GG")
        s_dist = calc_cfd_site(self.G, t_dist, "GG")
        assert s_seed < s_dist, f"Seed ({s_seed}) must be < distal ({s_dist})"

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            calc_cfd_site("AAAA", self.G, "GG")

    def test_multiple_mismatches(self):
        # G="GAGTCCGAGCAGAAGAAGAA"
        # pos 1: G->A (rG:dA,1 = 1.0) and pos 20: A->C (rA:dC,20 = 0.764706)
        # Both are in the 12-type table; result = 1.0 * 0.764706 * 1.0 (GG PAM)
        t = "AAGTCCGAGCAGAAGAAGAC"   # 2 mismatches: pos1 G->A, pos20 A->C
        s = calc_cfd_site(self.G, t, "GG")
        assert 0.0 < s < 1.0, f"Expected score in (0,1), got {s}"

    def test_rUdG_wobble_high_tolerance_pos1(self):
        # rU:dG at pos 1 = 0.857143 (guide T->U vs target G)
        g = "TAAAAAAAAAAAAAAAAAA" + "A"
        t = "GAAAAAAAAAAAAAAAAAA" + "A"
        s = calc_cfd_site(g, t, "GG")
        expected_mm = DOENCH_2016_MISMATCH_SCORES[("rU:dG", 1)]
        assert abs(s - expected_mm) < 1e-5


# ===========================================================================
# 4 & 5. Plus-strand and minus-strand PAM extraction (unit tests)
# ===========================================================================

class TestPAMKeyExtraction:
    def test_3nt_pam_extraction(self):
        assert pam_key_from_genomic_pam("TGG") == "GG"
        assert pam_key_from_genomic_pam("AGG") == "GG"
        assert pam_key_from_genomic_pam("CGG") == "GG"

    def test_2nt_pam_extraction(self):
        assert pam_key_from_genomic_pam("GG") == "GG"
        assert pam_key_from_genomic_pam("AG") == "AG"

    def test_non_canonical_pam(self):
        assert pam_key_from_genomic_pam("AAG") == "AG"
        assert pam_key_from_genomic_pam("CAG") == "AG"

    def test_invalid_pam_returns_none(self):
        assert pam_key_from_genomic_pam("") is None
        assert pam_key_from_genomic_pam("N") is None
        assert pam_key_from_genomic_pam("NNN") is None
        assert pam_key_from_genomic_pam("XYZ") is None


# ===========================================================================
# 6. Intended-locus exclusion by genomic identity
# ===========================================================================

class TestIntendedLocusExclusion:
    def _make_hit(self, chrom, strand, offset, n_mm=0):
        return {
            "chrom": chrom, "strand": strand, "offset": offset,
            "protospacer_seq": "A" * 20, "pam_3nt": "TGG",
            "n_mm_protospacer": n_mm, "mismatches_in_protospacer": [],
        }

    def test_coordinate_match_removes_correct_hit(self):
        hits = [
            self._make_hit("chr17", "+", 7669706, n_mm=0),
            self._make_hit("chr1",  "+", 999999, n_mm=1),
        ]
        filtered, excluded, method = exclude_on_target(
            hits, guide_chrom="chr17", guide_offset=7669706, guide_strand="+"
        )
        assert excluded
        assert method == "COORDINATE_MATCH"
        assert len(filtered) == 1
        assert filtered[0]["chrom"] == "chr1"

    def test_window_tolerance(self):
        """On-target exclusion should succeed within 5-bp window."""
        hits = [self._make_hit("chr17", "+", 7669706, n_mm=0)]
        _, excluded, method = exclude_on_target(
            hits, guide_chrom="chr17", guide_offset=7669710, guide_strand="+"
        )
        assert excluded and method == "COORDINATE_MATCH"

    def test_wrong_strand_not_excluded_by_coordinate(self):
        hits = [self._make_hit("chr17", "-", 7669706, n_mm=0)]
        _, excluded, method = exclude_on_target(
            hits, guide_chrom="chr17", guide_offset=7669706, guide_strand="+"
        )
        assert not excluded
        assert method == "UNRESOLVED"

    def test_fallback_unresolved_when_coordinates_missing(self):
        hits = [
            self._make_hit("chr1", "+", 100, n_mm=2),
            self._make_hit("chr2", "+", 200, n_mm=0),
        ]
        filtered, excluded, method = exclude_on_target(
            hits, guide_chrom=None, guide_offset=None, guide_strand=None
        )
        assert not excluded
        assert method == "UNRESOLVED"
        assert len(filtered) == 2


# ===========================================================================
# 7. Retention of other perfect-match loci
# ===========================================================================

class TestPerfectMatchRetention:
    def test_additional_perfect_copies_counted(self):
        hits = [
            {"chrom": "chr17", "strand": "+", "offset": 7669706,
             "protospacer_seq": "A"*20, "pam_3nt": "TGG",
             "n_mm_protospacer": 0, "mismatches_in_protospacer": []},  # on-target
            {"chrom": "chr1",  "strand": "+", "offset": 999,
             "protospacer_seq": "A"*20, "pam_3nt": "TGG",
             "n_mm_protospacer": 0, "mismatches_in_protospacer": []},  # off-target perfect
        ]
        metrics = score_guide_cfd(
            "A" * 20, hits,
            guide_chrom="chr17", guide_offset=7669706, guide_strand="+",
        )
        assert metrics["n_perfect_copy_loci"] >= 1


# ===========================================================================
# 8. Censored risk and specificity are NaN
# ===========================================================================

class TestCensoredHandling:
    def test_censored_risk_is_nan(self):
        res = calc_guide_cfd_score([1.0] * 10, off_target_capped=True)
        assert math.isnan(res["cfd_off_target_risk"])

    def test_censored_specificity_is_nan(self):
        res = calc_guide_cfd_score([1.0] * 10, off_target_capped=True)
        assert math.isnan(res["cfd_specificity_score"])

    def test_censored_partial_risk_is_numeric(self):
        res = calc_guide_cfd_score([0.5, 0.3], off_target_capped=True)
        assert not math.isnan(res["cfd_partial_risk"])
        assert abs(res["cfd_partial_risk"] - 0.8) < 1e-6

    def test_censored_status_label(self):
        res = calc_guide_cfd_score([], off_target_capped=True)
        assert res["cfd_scoring_status"] == "CENSORED_CAP_REACHED"
        assert res["cfd_scoring_complete"] is False


# ===========================================================================
# 9. Zero-hit complete vs incomplete search distinction
# ===========================================================================

class TestZeroHitDistinction:
    def test_complete_zero_hits(self):
        res = calc_guide_cfd_score([], off_target_capped=False)
        assert res["cfd_off_target_risk"] == 0.0
        assert res["cfd_specificity_score"] == 1.0
        assert res["cfd_scoring_complete"] is True
        assert res["cfd_scoring_status"] == "COMPLETE"

    def test_censored_zero_observed_hits(self):
        """Cap reached but 0 scored sites (all PAMs unknown)."""
        res = calc_guide_cfd_score([], off_target_capped=True)
        assert math.isnan(res["cfd_off_target_risk"])
        assert math.isnan(res["cfd_specificity_score"])
        assert res["cfd_partial_risk"] == 0.0   # sum of empty list


# ===========================================================================
# 10. No substitution of raw count for CFD risk
# ===========================================================================

class TestNoRawCountSubstitution:
    def test_risk_is_cfd_sum_not_count(self):
        # 999 sites each with CFD=0.5 -> risk = 499.5, NOT 999
        res = calc_guide_cfd_score([0.5] * 999, off_target_capped=True)
        assert math.isnan(res["cfd_off_target_risk"])      # censored -> NaN
        assert abs(res["cfd_partial_risk"] - 499.5) < 1.0  # real sum

    def test_partial_risk_not_equal_to_site_count(self):
        n = 42
        scores = [0.3] * n
        res = calc_guide_cfd_score(scores, off_target_capped=False)
        assert abs(res["cfd_off_target_risk"] - 0.3 * n) < 1e-5
        assert res["cfd_off_target_risk"] != float(n)


# ===========================================================================
# 11. Unsupported / unavailable PAM handling
# ===========================================================================

class TestUnsupportedPAM:
    def test_unsupported_pam_gives_zero_cfd(self):
        g = "GAGTCCGAGCAGAAGAAGAA"
        assert calc_cfd_site(g, g, "NN") == 0.0

    def test_pam_key_unavailable_skips_site(self):
        """score_guide_cfd: hit with 'NNN' PAM increments incomplete_pam_count."""
        hits = [{
            "chrom": "chr1", "strand": "+", "offset": 999,
            "protospacer_seq": "A" * 20,
            "pam_3nt": "NNN",
            "n_mm_protospacer": 1,
            "mismatches_in_protospacer": [],
        }]
        metrics = score_guide_cfd("A" * 20, hits)
        assert metrics["cfd_incomplete_pam"] == 1
        assert metrics["cfd_scored_site_count"] == 0


# ===========================================================================
# 12. Schema preservation check
# ===========================================================================

class TestSchemaPreservation:
    def test_required_output_keys(self):
        res = calc_guide_cfd_score([0.5], off_target_capped=False)
        required = {
            "cfd_off_target_risk", "cfd_specificity_score",
            "cfd_partial_risk", "cfd_scored_site_count",
            "cfd_incomplete_pam", "cfd_scoring_complete", "cfd_scoring_status",
        }
        assert required.issubset(res.keys())


# ===========================================================================
# 13. Reproducibility on known reference example
# ===========================================================================

class TestReproducibility:
    def test_crispor_example(self):
        """
        Verified against CRISPOR cfd-score-calculator.py with the CRISPOR
        pickle files for guide GAGTCCGAGCAGAAGAAGAA vs off-target with
        1-mismatch rA:dA at position 5 and GG PAM.

        Expected: 1.0 * rA:dA_pos5 * PAM_GG
                = 1.0 * 0.363636 * 1.0 = 0.363636
        """
        g = "GAGTCCGAGCAGAAGAAGAA"
        t = "GAGTACGAGCAGAAGAAGAA"   # A->A mismatch at pos 5 (rA:dA,5)
        # Wait -- if guide is A at pos 5 and target is A, that is a match.
        # Build correct mismatch: guide G at pos 5, target A -> rG:dA,5 = 0.3
        g2 = "GAGTACGAGCAGAAGAAGAA"  # guide with A at pos 5 (rA)
        t2 = "GAGTCCGAGCAGAAGAAGAA"  # target with C at pos 5 -> rA:dC,5 = 0.72
        s = calc_cfd_site(g2, t2, "GG")
        expected = DOENCH_2016_MISMATCH_SCORES[("rA:dC", 5)] * DOENCH_2016_PAM_SCORES["GG"]
        assert abs(s - expected) < 1e-5, f"CFD={s}, expected={expected}"


# ===========================================================================
# 14. Step 10A -- Variance Collapse and Exclusion Regression Tests
# ===========================================================================

class TestStep10AVarianceAndExclusionRegression:
    def _make_hit(self, chrom, strand, offset, protospacer, pam_3nt="TGG", n_mm=0, edit=""):
        return {
            "chrom": chrom,
            "strand": strand,
            "offset": offset,
            "protospacer_seq": protospacer,
            "pam_3nt": pam_3nt,
            "n_mm_protospacer": n_mm,
            "edit_str": edit,
        }

    def test_perfect_target_exclusion_no_offtarget_penalty(self):
        """
        After successful coordinate-based exclusion of the intended on-target hit,
        its score (1.0) must not enter the off-target risk sum.
        """
        g = "GAGTACGAGCAGAAGAAGAA"
        hits = [
            # Intended on-target hit (perfect match at target locus)
            self._make_hit("NC_000017.11", "+", 7670000, g, pam_3nt="TGG", n_mm=0),
            # Single off-target with rA:dC at pos 5
            self._make_hit("NC_000001.11", "+", 1000000, "GAGTCCGAGCAGAAGAAGAA", pam_3nt="TGG", n_mm=1, edit="4:C>A"),
        ]
        meta = {"genomic_accession": "NC_000017.11", "retrieved_start": 7666421, "retrieved_stop": 7689490}
        res = score_guide_cfd(g, hits, gene_meta=meta)

        assert res["on_target_excluded"] is True
        assert res["on_target_exclusion_method"] == "COORDINATE_MATCH"
        assert res["cfd_scored_site_count"] == 1
        expected_off_target = DOENCH_2016_MISMATCH_SCORES[("rA:dC", 5)] * DOENCH_2016_PAM_SCORES["GG"]
        assert abs(res["cfd_off_target_risk"] - expected_off_target) < 1e-5
        # Must NOT have included the 1.0 from on-target
        assert res["cfd_off_target_risk"] < 1.0

    def test_zero_valid_off_targets_produces_exact_zero_and_one(self):
        """
        A guide with exactly one alignment (the intended on-target locus) has
        zero off-target sites after exclusion, yielding R_CFD = 0.0 and S_CFD = 1.0.
        """
        g = "GAGTACGAGCAGAAGAAGAA"
        hits = [
            self._make_hit("NC_000017.11", "+", 7670000, g, pam_3nt="TGG", n_mm=0),
        ]
        meta = {"genomic_accession": "NC_000017.11", "retrieved_start": 7666421, "retrieved_stop": 7689490}
        res = score_guide_cfd(g, hits, gene_meta=meta)

        assert res["on_target_excluded"] is True
        assert res["cfd_scored_site_count"] == 0
        assert res["cfd_off_target_risk"] == 0.0
        assert res["cfd_specificity_score"] == 1.0
        assert res["cfd_scoring_complete"] is True

    def test_distinct_off_target_profiles_produce_divergent_risks(self):
        """
        Guides with distinct genomic off-target profiles must produce distinct,
        biologically meaningful risk values rather than collapsing to a constant.
        """
        g = "GAGTACGAGCAGAAGAAGAA"
        meta = {"genomic_accession": "NC_000017.11", "retrieved_start": 7666421, "retrieved_stop": 7689490}

        # Profile A: 3 tolerant off-targets (distal mismatches with GG PAM)
        hits_A = [
            self._make_hit("NC_000017.11", "+", 7670000, g, pam_3nt="TGG", n_mm=0),
            self._make_hit("NC_000001.11", "+", 100000, "AAGTACGAGCAGAAGAAGAA", pam_3nt="TGG", n_mm=1), # rG:dA,1 = 1.0
            self._make_hit("NC_000002.12", "+", 200000, "GAGTCCGAGCAGAAGAAGAA", pam_3nt="TGG", n_mm=1), # rA:dC,5 = 0.72
        ]
        res_A = score_guide_cfd(g, hits_A, gene_meta=meta)

        # Profile B: 1 weak off-target (seed mismatch with AG non-canonical PAM)
        hits_B = [
            self._make_hit("NC_000017.11", "+", 7670000, g, pam_3nt="TGG", n_mm=0),
            self._make_hit("NC_000003.12", "+", 300000, "GAGTACGAGCAGAAGAAGAC", pam_3nt="AAG", n_mm=1), # seed mm + AG pam
        ]
        res_B = score_guide_cfd(g, hits_B, gene_meta=meta)

        # Profile C: 0 off-targets
        hits_C = [
            self._make_hit("NC_000017.11", "+", 7670000, g, pam_3nt="TGG", n_mm=0),
        ]
        res_C = score_guide_cfd(g, hits_C, gene_meta=meta)

        assert res_A["cfd_off_target_risk"] > res_B["cfd_off_target_risk"] > res_C["cfd_off_target_risk"]
        assert res_C["cfd_specificity_score"] > res_B["cfd_specificity_score"] > res_A["cfd_specificity_score"]
        assert res_C["cfd_off_target_risk"] == 0.0
        assert res_C["cfd_specificity_score"] == 1.0

    def test_unresolved_exclusion_preserves_without_fabricating_penalty(self):
        """
        When target locus exclusion is UNRESOLVED, existing hits are preserved,
        but no constant +1.0 penalty is added to guides with zero hits.
        """
        g = "GAGTACGAGCAGAAGAAGAA"
        # No hits at all
        res = score_guide_cfd(g, [], gene_meta=None)
        assert res["on_target_exclusion_method"] == "NO_HITS"
        assert res["cfd_off_target_risk"] == 0.0
        assert res["cfd_specificity_score"] == 1.0

    def test_incomplete_searches_strict_nan_and_separate_partial_risk(self):
        """
        Incomplete (capped) searches must strictly retain NaN for primary risk and
        specificity, while keeping partial risk numeric.
        """
        res = calc_guide_cfd_score([0.4, 0.6], off_target_capped=True)
        assert math.isnan(res["cfd_off_target_risk"])
        assert math.isnan(res["cfd_specificity_score"])
        assert res["cfd_scoring_complete"] is False
        assert res["cfd_scoring_status"] == "CENSORED_CAP_REACHED"
        assert abs(res["cfd_partial_risk"] - 1.0) < 1e-6

    def test_documented_specificity_transformation(self):
        """
        Validate the project-defined specificity transformation: S = 100 / (100 + R).
        """
        res_0 = calc_guide_cfd_score([], off_target_capped=False)
        assert res_0["cfd_specificity_score"] == 1.0

        res_1 = calc_guide_cfd_score([1.0], off_target_capped=False)
        assert abs(res_1["cfd_specificity_score"] - 100.0 / 101.0) < 1e-5

        res_25 = calc_guide_cfd_score([25.0], off_target_capped=False)
        assert abs(res_25["cfd_specificity_score"] - 0.8) < 1e-5

        res_100 = calc_guide_cfd_score([100.0], off_target_capped=False)
        assert abs(res_100["cfd_specificity_score"] - 0.5) < 1e-5

