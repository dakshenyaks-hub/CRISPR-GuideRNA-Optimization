# Step 9C — Final Defect Correction and Validation Report

## 1. Files Modified and Defects Corrected
- **`src/step9c_cfd.py`**: Modified `exclude_on_target()` to remove the unsafe `FIRST_ZEROMM_FALLBACK`. Alignments that do not match the expected coordinate window are no longer silently excluded. Instead, if coordinate matching fails, the function returns `method = "UNRESOLVED"` and preserves all 0-mismatch alignments to prevent accidental exclusion of genuine multi-copy loci.
- **`tests/test_cfd_e2e.py`**: Created a new end-to-end test suite that directly interfaces with the local GRCh38 FASTA via `pyfaidx` and tests real Bowtie index queries without fabricating data.

## 2. Unit and End-to-End Test Results
- All unit tests in `test_cfd_score.py` continue to pass.
- End-to-end extraction tests in `test_cfd_e2e.py` successfully retrieved GRCh38 loci, passing actual 20-nt coordinates through the extraction bounds and retrieving the appropriate PAM.
- **Result:** PASS. Execution evidence available in background task logs. 

## 3. Evidence for Both-Strand PAM Extraction
- **End-to-End Test:** `test_actual_grch38_extraction()` explicitly queries Bowtie and the genome index. The logic `pam_3nt = revcomp(genome[chrom][offset - 3 : offset].seq.upper())` for minus-strand targets mathematically returns the 5' (on positive strand) -> 3' (on negative strand) PAM and was verified against actual GRCh38 FASTA outputs.
- **Result:** PASS. 

## 4. Intended-Target Exclusion Test Results
- **End-to-End Test:** `test_exclude_on_target_unresolved()` verified that passing in 0-mismatch hits without sufficient coordinate data correctly returns `excluded=False` and `method="UNRESOLVED"`, retaining the loci rather than inappropriately deleting the first one. 
- **Result:** PASS. 

## 5. Reconciliation Statistics for 4,457 Suspicious Guides
- **Status:** UNRESOLVED.
- **Evidence:** Attempts were made to isolate exactly 4,457 guides based on zero CFD risk, capped off-target counts, missing PAM sites, or boolean flags across `step9_cfd` and `step9_cfd_corrected` artifacts. No single criteria mapped cleanly to a 4,457 count. Without a definitive manifest of these 4,457 specific guides, a robust guide-by-guide reconciliation against underlying alignment records cannot be completed without fabricating results.

## 6. Formula Provenance
- **Site-level calculation:** Multiplicative product of positional mismatch penalty and PAM penalty exactly mirrors the published Doench 2016 model.
- **Risk Summation:** `cfd_partial_risk` is the sum of site scores, mirroring published aggregation.
- **Project-Level Normalization:** `cfd_specificity_score = 100 / (100 + r_cfd)` is a project-defined convention adapted from the CRISPOR web tool to normalize raw risk sums into a bounded percentage-like scale (it is not part of the original Doench 2016 paper, which only provided the risk sum).
- **Incomplete Searches:** Capped searches correctly set final risk and specificity to `NaN`, preventing numeric leakage, while retaining the partial sum in `cfd_partial_risk`.

---

## Conclusion

NOT READY FOR STEP 9C RECOMPUTATION

**Remaining Limitations:**
- The 4,457 suspicious guides must be explicitly provided in a manifest or their filtering criteria unambiguously defined before guide-by-guide reconciliation can be completed.
