# Step 9C — Corrected CFD Dataset Audit and Readiness Assessment

## 1. Executive Summary
A comprehensive audit of the corrected Cutting-Frequency Determination (CFD) dataset (`results/step9_cfd_corrected/cfd_guide_features.csv`) and pipeline implementation was conducted. The audit verified data integrity, intended-target exclusion mechanics, CFD scoring accuracy, unit and end-to-end test execution, and downstream optimization readiness. 

While the underlying algorithm in `src/step9c_cfd.py` has now been patched and verified via 14 passing unit/E2E tests, the existing CSV dataset on disk contains legacy artifacts (such as 100% usage of `FIRST_ZEROMM_FALLBACK` prior to the patch and status flag misclassifications for 10,081 capped guides). Therefore, the dataset requires a targeted recomputation pass with the updated code before downstream optimization can safely execute.

**Final Status:** `READY AFTER SPECIFIC FIXES`

---

## 2. Files and Dataset Versions Inspected
- `src/cfd_score.py`: Authoritative scoring engine containing `DOENCH_2016_MISMATCH_SCORES` (240 entries), `DOENCH_2016_PAM_SCORES` (16 entries), and `calc_guide_cfd_score()`.
- `src/step9c_cfd.py`: Core pipeline implementation containing coordinate extraction, intended-target exclusion (`exclude_on_target`), and dataset aggregation.
- `tests/test_cfd_score.py`: 12 unit tests verifying mismatch tables, PAM tables, site aggregation, NaN handling for capped guides, and edge cases.
- `tests/test_cfd_e2e.py`: 2 end-to-end tests validating real GRCh38 FASTA retrieval via `pyfaidx`, Bowtie index alignments, both-strand PAM extraction, and coordinate fallback logic.
- `tests/run_all_tests.py`: Consolidated test runner executing the complete suite of 14 unit and E2E tests.
- `results/step9_cfd_corrected/cfd_guide_features.csv`: Primary dataset containing 59,361 candidate gRNA feature records.
- `reports/step9c_final_audit/step9c_final_report.md` & `original_group_recovery.md`: Prior audit reports.

---

## 3. Dataset Integrity Statistics
- **Total Guides:** 59,361
- **Unique Guide IDs:** 59,361 (0 duplicate IDs)
- **Missing Values:**
  - `deep_spcas9_score`: 7 missing
  - `cfd_off_target_risk`: 19,981 missing (`NaN`)
  - `cfd_specificity_score`: 19,981 missing (`NaN`)
- **Complete vs Incomplete Searches:**
  - **Complete (`cfd_scoring_complete = True`):** 39,380 guides (66.34%)
  - **Incomplete / Censored (`cfd_scoring_complete = False`):** 19,981 guides (33.66%)
- **Bowtie Hit Cap Handling:**
  - `CENSORED_CAP_REACHED` in Step 9C: 19,981 guides
  - `off_target_capped == True` from Step 8: 30,062 guides
- **PAM Availability & Integrity:**
  - Guides with incomplete PAM information (`cfd_incomplete_pam > 0`): 22,070 guides
  - Total skipped PAM sites across all guides: 14,910,696 sites
- **Intended-Target Exclusion Methods (in existing CSV):**
  - `FIRST_ZEROMM_FALLBACK`: 59,361 guides (100% of existing CSV records were computed prior to the `exclude_on_target` code patch)
- **Flag and Score Inconsistencies Identified:**
  - **Inconsistency 1:** 10,081 guides were marked as `off_target_capped == True` in Step 8 (reaching 1,000 Bowtie hits), yet were marked as `cfd_scoring_complete == True` (`COMPLETE`) in the existing CSV.
  - **Inconsistency 2:** 19,981 guides have `NaN` for primary risk/specificity. If passed unfiltered into Pareto optimization, these `NaN` values will break distance calculations or lead to invalid front ranking.

---

## 4. Intended-Target Exclusion Validation
- **Coordinate Matching:** `exclude_on_target` in `src/step9c_cfd.py` was updated to remove `FIRST_ZEROMM_FALLBACK`. It matches hits within a 5-bp window using chromosome, strand, and offset.
- **Unresolved Exclusions:** If coordinate matching fails, the function now returns `method = "UNRESOLVED"` and retains all hits rather than silently deleting the first zero-mismatch hit.
- **Strand & Orientation Correctness:** Tested in `test_actual_grch38_extraction` and `test_exclude_on_target_unresolved`. Plus-strand PAMs are retrieved downstream of offset (`+20` to `+23`), while minus-strand PAMs are reverse-complemented upstream (`-3` to `0`).

---

## 5. CFD Scoring Validation
- **Penalty Tables:** 240 mismatch penalties and 16 PAM penalties match the CRISPOR reference data with 0 discrepancies.
- **Site-Level Aggregation:** Multiplicative product of positional mismatch penalty and PAM penalty strictly matches Doench et al. (2016).
- **Censored Searches:** When `off_target_capped = True`, `cfd_off_target_risk` and `cfd_specificity_score` are strictly set to `NaN`, preventing capped guides from falsely appearing high-specificity. Partial risk is preserved in `cfd_partial_risk`.
- **Normalization Provenance:** The specificity score formula `S_CFD = 100 / (100 + R_CFD)` is explicitly documented as a project-defined convention adapted from CRISPOR, not a published formula from the original 2016 paper.

---

## 6. Test Commands and Actual Results
The entire test suite was executed via `python tests/run_all_tests.py`:

```
=== Running Unit Tests in tests/test_cfd_score.py ===
  [PASS] TestMismatchTable
  [PASS] TestPAMTable
  [PASS] TestCalcCFDSite
  [PASS] TestPAMKeyExtraction
  [PASS] TestIntendedLocusExclusion
  [PASS] TestPerfectMatchRetention
  [PASS] TestCensoredHandling
  [PASS] TestZeroHitDistinction
  [PASS] TestNoRawCountSubstitution
  [PASS] TestUnsupportedPAM
  [PASS] TestSchemaPreservation
  [PASS] TestReproducibility

=== Running E2E Tests in tests/test_cfd_e2e.py ===
  [PASS] test_actual_grch38_extraction
  [PASS] test_exclude_on_target_unresolved

ALL UNIT AND E2E TESTS PASSED SUCCESSFULLY!
```

---

## 7. Optimization Readiness Assessment
The dataset is **not ready in its current CSV state**, but will be **ready after specific fixes**:
1. **Selection Bias & NaN Handling:** 33.66% (19,981) of guides have `NaN` risk scores due to hit capping. Optimization algorithms (NSGA-II, weighted sum, convex optimization) must explicitly filter out `cfd_scoring_complete == False` records or handle missing values to avoid corrupting the Pareto front.
2. **CSV Regeneration Needed:** Because all 59,361 rows in `results/step9_cfd_corrected/cfd_guide_features.csv` were generated before the patch to `src/step9c_cfd.py`, the CSV file still contains `FIRST_ZEROMM_FALLBACK` for all rows and 10,081 misclassified capped searches.

---

## 8. Remaining Limitations
- 10,081 guides that hit the 1,000 Bowtie alignment cap require re-evaluation to ensure `cfd_scoring_complete = False` is consistently applied.
- 22,070 guides have at least 1 off-target site where genomic PAM sequence could not be retrieved (`cfd_incomplete_pam > 0`).

---

## 9. Exact Recommended Next Action
1. Execute a clean, lightweight recomputation script to regenerate `results/step9_cfd_corrected/cfd_guide_features.csv` using the patched `src/step9c_cfd.py` code.
2. Update downstream optimization scripts to explicitly filter `cfd_scoring_complete == True` before running Pareto optimization.

---

## Final Status

`READY AFTER SPECIFIC FIXES`
