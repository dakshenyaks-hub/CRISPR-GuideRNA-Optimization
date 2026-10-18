# Step 9C -- CFD Correction Report

## Corrections Applied

| # | Issue | Fix |
|---|---|---|
| A | Mismatch table was synthetic approximation | Replaced with exact CRISPOR pickle values |
| B | PAM hardcoded as AGG | 23-mer Bowtie alignment; PAM from read_seq[20:23] |
| C | Censored scores stored as numbers | NaN for cfd_off_target_risk and cfd_specificity_score |
| D | On-target exclusion was positional | Coordinate-based match with 5-nt window |

## Source Provenance

  Mismatch table: maximilianh/crisporWebsite CFD_Scoring/mismatch_score.pkl (blob SHA 18f9f169d00f377a2704ac51fa57a9571c571bbb)
  PAM table:      maximilianh/crisporWebsite CFD_Scoring/pam_scores.pkl (blob SHA 86068fb0ac99de7fff296bf7181c327e83a39a9c)
  License:        MIT
  Reference:      Doench JG et al. Nat Biotechnol 34:184-191 (2016)

## Validation Results

  13/14 checks passed.

 check_id                                               description  passed                                                                  details
        1          Exact mismatch penalties match CRISPOR reference    True rA:dA,pos1=1.000000 (expect 1.0); rU:dG,pos20=0.090909 (expect 0.090909)
        2               Exact PAM penalties match CRISPOR reference    True             GG=1.0 (1.0); AG=0.259259 (0.259259); GA=0.069444 (0.069444)
        3                 Mismatch table completeness (240 entries)    True                                                             Entries: 240
        4                              Perfect match + GG PAM = 1.0    True                                                             CFD_site=1.0
        5                         Non-canonical AG PAM scores 0.259    True                        CFD_site with AG PAM = 0.259259 (expect 0.259259)
        6                      Unknown PAM (NN not in table) -> 0.0    True                                               CFD_site with NN PAM = 0.0
        7                  Seed mismatch penalised more than distal   False                                      seed_CFD=0.7647 < distal_CFD=0.0000
        8                       R_CFD aggregation (0.5 + 0.3 = 0.8)    True                                                              R_CFD = 0.8
        9              Censored guide: risk and specificity are NaN    True                                                       risk=nan, spec=nan
       10     Censored guide: cfd_partial_risk is numeric (not NaN)    True                                                     cfd_partial_risk=5.0
       11              Complete guide with 0 off-targets: S_CFD=1.0    True                                                     S_CFD=1.0, R_CFD=0.0
       12 pam_key_from_genomic_pam: TGG->GG, AAG->AG, invalid->None    True                                 TGG->GG, AAG->AG, empty->None, NNN->None
       13                      Coordinate-based on-target exclusion    True                      excluded=True, method=COORDINATE_MATCH, remaining=1
       14       cfd_partial_risk is a real CFD sum, not a raw count    True                 partial_risk=5.0 (5 * CFD_site=1.0 = 5.0, not raw count)

## Per-Gene Summary (complete CFD scores only)

 gene  total_candidates  complete_cfd_scoring  censored_cfd_scoring  missing_cfd_risk_scores  n_sites_with_missing_pam  n_perfect_copy_loci_total  mean_cfd_off_target_risk  min_cfd_off_target_risk  max_cfd_off_target_risk  mean_cfd_specificity_score  min_cfd_specificity_score  max_cfd_specificity_score
 TP53              4398                  2937                  1461                     1461                   1065276                     499428                       0.0                      0.0                      0.0                         1.0                        1.0                        1.0
  HBB               540                   520                    20                       20                     14743                       8323                       0.0                      0.0                      0.0                         1.0                        1.0                        1.0
 EGFR             23934                 23305                   629                      629                    465606                     288844                       0.0                      0.0                      0.0                         1.0                        1.0                        1.0
BRCA1             30489                 12618                 17871                    17871                  13365071                    5145462                       0.0                      0.0                      0.0                         1.0                        1.0                        1.0