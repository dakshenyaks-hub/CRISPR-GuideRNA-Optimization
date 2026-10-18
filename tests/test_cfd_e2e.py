import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.step9c_cfd import parse_bowtie_and_extract_cfd_features, get_genome_fasta, exclude_on_target

def test_actual_grch38_extraction():
    try:
        genome = get_genome_fasta()
    except FileNotFoundError:
        print("GRCh38 genome not available, skipping end-to-end extraction test")
        return

    guides = ["GAGTCCGAGCAGAAGAAGAA", "TCCCTGGAGACCGTCATCCG"]
    index_prefix = str(PROJECT_ROOT / "data" / "hg38_index" / "hg38_index")
    
    if not os.path.exists(index_prefix + ".1.ebwt"):
        print("Bowtie index not available")
        return
        
    hits = parse_bowtie_and_extract_cfd_features(guides, index_prefix, genome, max_mismatches=0)
    assert len(hits) == 2, "Hits length should be 2"
    for g, hlist in hits.items():
        for hit in hlist:
            assert len(hit["protospacer_seq"]) == 20
            if hit["chrom"] in genome:
                assert hit["pam_3nt"] != ""

def test_exclude_on_target_unresolved():
    hits = [
        {"chrom": "chr1", "strand": "+", "offset": 100, "n_mm_protospacer": 0},
        {"chrom": "chr2", "strand": "-", "offset": 200, "n_mm_protospacer": 1},
    ]
    filtered, excluded, method = exclude_on_target(hits)
    assert not excluded, "Should not be excluded"
    assert method == "UNRESOLVED", f"Method was {method}"
    assert len(filtered) == 2, "Should retain both hits"

if __name__ == "__main__":
    test_actual_grch38_extraction()
    test_exclude_on_target_unresolved()
    print("All E2E tests passed.")

