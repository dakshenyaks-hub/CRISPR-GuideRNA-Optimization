import json
import logging
import math
import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cfd_score import calc_guide_cfd_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("regenerate_step9_final")

def main():
    in_csv = PROJECT_ROOT / "results" / "step9_cfd_corrected" / "cfd_guide_features.csv"
    out_dir = PROJECT_ROOT / "results" / "step9_cfd_final"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading input dataset from {in_csv}...")
    df = pd.read_csv(in_csv)
    log.info(f"Loaded {len(df)} candidate guides.")

    # Load gene metadata
    gene_metas = {}
    for g in ["TP53", "HBB", "EGFR", "BRCA1"]:
        matched = list((PROJECT_ROOT / "data" / "reference_sequences").glob(f"{g}_*.metadata.json"))
        if matched:
            with open(matched[0], "r") as fh:
                gene_metas[g] = json.load(fh)
                log.info(f"Loaded {g} metadata: Chrom {gene_metas[g].get('genomic_accession')}, Span [{gene_metas[g].get('retrieved_start')}, {gene_metas[g].get('retrieved_stop')}]")

    df_final = df.copy()

    # 1. Correct capped search status and set primary scores to NaN
    capped_mask = df_final["off_target_capped"].astype(bool)
    df_final.loc[capped_mask, "cfd_scoring_complete"] = False
    df_final.loc[capped_mask, "cfd_scoring_status"] = "CENSORED_CAP_REACHED"
    df_final.loc[capped_mask, "cfd_off_target_risk"] = float("nan")
    df_final.loc[capped_mask, "cfd_specificity_score"] = float("nan")

    # 2. Correct intended-target exclusion method
    exclusion_methods = []
    on_target_excluded_list = []
    
    # Recalculate scores for uncapped guides if target exclusion changes
    new_risk_list = df_final["cfd_off_target_risk"].tolist()
    new_spec_list = df_final["cfd_specificity_score"].tolist()
    new_partial_risk_list = df_final["cfd_partial_risk"].tolist()

    for idx, row in df_final.iterrows():
        gene = row["gene"]
        meta = gene_metas.get(gene, {})
        start_coord = meta.get("retrieved_start")
        stop_coord = meta.get("retrieved_stop")
        
        pos = row["pam_position"]
        is_coord_match = False
        if start_coord is not None and stop_coord is not None and pd.notna(pos):
            min_pos = min(start_coord, stop_coord) - 200
            max_pos = max(start_coord, stop_coord) + 200
            if min_pos <= pos <= max_pos:
                is_coord_match = True
                
        if is_coord_match:
            method = "COORDINATE_MATCH"
            excluded = True
        else:
            method = "UNRESOLVED"
            excluded = False
            
        exclusion_methods.append(method)
        on_target_excluded_list.append(excluded)

        # If uncapped and on-target was previously excluded by fallback but now UNRESOLVED,
        # we adjust partial risk and scores to include the retained 0-mm on-target hit (+1.0 risk)
        is_capped = row["off_target_capped"]
        prev_method = row.get("on_target_exclusion_method", "")
        if not is_capped:
            if prev_method == "FIRST_ZEROMM_FALLBACK" and not excluded:
                # Retain on-target hit in partial risk
                cur_p_risk = row["cfd_partial_risk"]
                adj_p_risk = cur_p_risk + 1.0 if pd.notna(cur_p_risk) else 1.0
                new_partial_risk_list[idx] = adj_p_risk
                new_risk_list[idx] = adj_p_risk
                new_spec_list[idx] = round(100.0 / (100.0 + adj_p_risk), 6)

    df_final["on_target_exclusion_method"] = exclusion_methods
    df_final["on_target_excluded"] = on_target_excluded_list
    df_final["cfd_partial_risk"] = new_partial_risk_list
    df_final["cfd_off_target_risk"] = new_risk_list
    df_final["cfd_specificity_score"] = new_spec_list

    # Save output CSV
    out_csv = out_dir / "cfd_guide_features.csv"
    df_final.to_csv(out_csv, index=False)
    log.info(f"Saved regenerated dataset to {out_csv}")

    # Generate per-gene summary records
    summary_records = []
    for g in ["TP53", "HBB", "EGFR", "BRCA1"]:
        gdf = df_final[df_final["gene"] == g]
        complete = gdf[gdf["cfd_scoring_complete"] == True]
        censored = gdf[gdf["cfd_scoring_complete"] == False]
        nan_risk_count = gdf["cfd_off_target_risk"].isna().sum()
        n_incomplete_pam = int(gdf["cfd_incomplete_pam"].sum())
        n_perfect_copies = int(gdf["n_perfect_copy_loci"].sum())

        summary_records.append({
            "gene": g,
            "total_candidates": len(gdf),
            "complete_cfd_scoring": len(complete),
            "censored_cfd_scoring": len(censored),
            "missing_cfd_risk_scores": int(nan_risk_count),
            "n_sites_with_missing_pam": n_incomplete_pam,
            "n_perfect_copy_loci_total": n_perfect_copies,
            "mean_cfd_off_target_risk": round(complete["cfd_off_target_risk"].mean(), 4) if len(complete) > 0 else float("nan"),
            "min_cfd_off_target_risk": round(complete["cfd_off_target_risk"].min(), 4) if len(complete) > 0 else float("nan"),
            "max_cfd_off_target_risk": round(complete["cfd_off_target_risk"].max(), 4) if len(complete) > 0 else float("nan"),
            "mean_cfd_specificity_score": round(complete["cfd_specificity_score"].mean(), 4) if len(complete) > 0 else float("nan"),
            "min_cfd_specificity_score": round(complete["cfd_specificity_score"].min(), 4) if len(complete) > 0 else float("nan"),
            "max_cfd_specificity_score": round(complete["cfd_specificity_score"].max(), 4) if len(complete) > 0 else float("nan"),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_dir / "cfd_scoring_summary.csv", index=False)
    log.info("Saved summary table to cfd_scoring_summary.csv")

    log.info("=" * 60)
    log.info(f"REGENERATION COMPLETE: {len(df_final)} total guides.")
    log.info(f"Complete searches: {(df_final['cfd_scoring_complete'] == True).sum()}")
    log.info(f"Incomplete (capped) searches: {(df_final['cfd_scoring_complete'] == False).sum()}")
    log.info(f"Exclusion methods: {df_final['on_target_exclusion_method'].value_counts().to_dict()}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
