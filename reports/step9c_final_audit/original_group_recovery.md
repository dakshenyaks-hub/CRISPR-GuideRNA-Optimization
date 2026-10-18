# Step 9C — Original 4,457-Guide Group Recovery Report

## 1. Search Performed for the Original 4,457 Guides
To recover the exact group of 4,457 suspicious guides referenced in previous requests, a comprehensive search of the entire repository was conducted:
1. **Ripgrep Search:** Queried the entire workspace (including untracked files in `results/`, `reports/`, and `data/`) for the strings `"4,457"` and `"4457"` across `*.csv`, `*.json`, `*.py`, `*.md`, and `*.txt` files.
2. **Git History Search:** Executed `git log -S "4457"`, `git log -S "4,457"`, and `git log -S "suspicious"` to examine all prior commits, reports, and code states for historical references to this group.
3. **Dataframe Auditing:** Wrote and executed Python scripts to parse `results/step9_cfd/cfd_guide_features.csv` and `results/step9_cfd_corrected/cfd_guide_features.csv`. The script iterated over every column (boolean, numerical, and categorical) to test if exactly 4,457 guides shared any condition (e.g., zero off-target risk, missing PAM count, `CENSORED_CAP_REACHED` status, etc.).

## 2. Search Results and Conclusion
- The string "4457" only appeared in the repository as genomic coordinate fragments (e.g., `EGFR_44572`) or trailing floating-point decimals (e.g., `55.764457`). There were zero whole-word occurrences indicating a group size or a documented audit finding.
- Git history searches returned absolutely no matches for the number or the word "suspicious."
- Dataframe auditing proved that the number of censored guides in `step9_cfd` was 14,048, and in the corrected `step9_cfd_corrected` run it was 19,981. There is no single column or status flag that isolates exactly 4,457 guides.

**Why the original group is unrecoverable:**
The number 4,457 was exclusively provided in external prompts and does not correspond to any preserved file, output artifact, git commit, or reproducible dataframe query within this workspace. Without an original intermediate dataset, report, or reproducible filter logic, the specific identities of these 4,457 guides cannot be mathematically reconstructed.

## 3. Independent Assessment of the Current Corrected Dataset
Despite the unrecoverability of the historical 4,457-guide group, the current corrected dataset (`results/step9_cfd_corrected/cfd_guide_features.csv`) **can be independently validated**. It has been structured to use explicit, reproducible rules rather than ambiguous "suspicious" flags:
- **Alignment Completeness:** Safely tracked via `cfd_scoring_complete` (True/False). If Bowtie hits are capped (>=1000), this flag is False, and the primary risk metrics are strictly set to `NaN`.
- **PAM Availability:** Explicitly quantified per guide in the `cfd_incomplete_pam` column, ensuring untestable loci are tracked rather than silently swallowed.
- **Intended-Target Exclusion:** The modified algorithm strictly relies on coordinate and gene-span bounding. The exact action taken is tracked via the boolean `on_target_excluded` and the string `on_target_exclusion_method` (either `"COORDINATE_MATCH"` or `"UNRESOLVED"`).

Because every guide explicitly carries the provenance of its extraction, alignment cap, and exclusion logic, the dataset is robustly verifiable without needing to reference the untraceable 4,457-guide group.

---

### Conclusion
ORIGINAL GROUP UNRECOVERABLE
