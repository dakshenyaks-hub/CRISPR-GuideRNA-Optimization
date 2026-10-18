"""
pam_guide_module.py

Module 1 of the Multi-Objective CRISPR Guide RNA Design Framework.
Responsibilities:
    - Read a DNA sequence from a FASTA file (Biopython).
    - Scan both strands for SpCas9 PAM motifs (NGG).
    - Extract the corresponding 20-nt guide RNA for each valid PAM.
    - Store candidates in a Pandas DataFrame and export to CSV.
    - Print a summary and a PAM-highlighted view of the sequence.

Author: Bioinformatics Development Team
"""

import re
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

GUIDE_LENGTH = 20
PAM_LENGTH = 3
PAM_REGEX = re.compile(r"(?=([ACGT]GG))")  # NGG, overlapping matches

# ANSI colors for console highlighting
_COLOR_PAM = "\033[91m"   # red
_COLOR_RESET = "\033[0m"


def read_fasta_sequence(filepath: str) -> str:
    """Read the first record of a FASTA file and return its sequence as a string."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")

    record = next(SeqIO.parse(str(path), "fasta"), None)
    if record is None:
        raise ValueError(f"No records found in FASTA file: {filepath}")

    return str(record.seq).upper()


def find_pam_sites(sequence: str, strand: str = "+") -> list[dict]:
    """
    Scan a single strand for NGG PAM motifs.
    Returns a list of dicts: {pam_start, pam_seq, strand}.
    pam_start is 0-based, relative to the given (strand) sequence.
    """
    sites = []
    for match in PAM_REGEX.finditer(sequence):
        pam_start = match.start()
        pam_seq = match.group(1)
        sites.append({"pam_start": pam_start, "pam_seq": pam_seq, "strand": strand})
    return sites


def extract_guide(sequence: str, pam_start: int) -> str | None:
    """
    Extract the 20-nt guide RNA immediately upstream (5') of a PAM.
    Returns None if there isn't enough upstream sequence (edge case).
    """
    guide_start = pam_start - GUIDE_LENGTH
    if guide_start < 0:
        return None  # not enough upstream bases -> invalid candidate
    return sequence[guide_start:pam_start]


def build_guide_dataframe(sequence: str) -> pd.DataFrame:
    """
    Scan forward and reverse strands, extract guides for every valid PAM,
    and return a consolidated DataFrame of candidates.
    """
    rev_comp = str(Seq(sequence).reverse_complement())
    seq_len = len(sequence)

    records = []
    for strand, strand_seq in (("+", sequence), ("-", rev_comp)):
        for site in find_pam_sites(strand_seq, strand=strand):
            guide = extract_guide(strand_seq, site["pam_start"])
            if guide is None:
                continue  # skip invalid edge cases

            # Map position back to original (+) strand coordinates for reporting
            if strand == "+":
                genomic_pos = site["pam_start"]
            else:
                genomic_pos = seq_len - site["pam_start"] - PAM_LENGTH

            records.append(
                {
                    "guide_rna": guide,
                    "pam_sequence": site["pam_seq"],
                    "pam_position": genomic_pos,
                    "strand": strand,
                    "guide_length": len(guide),
                }
            )

    df = pd.DataFrame(records).sort_values(
        by=["pam_position", "strand"]
    ).reset_index(drop=True)
    return df


def highlight_pam_in_sequence(sequence: str, df: pd.DataFrame) -> str:
    """
    Build a console-friendly string of the forward strand with PAM sites
    highlighted in red. Only forward-strand PAMs are highlighted in-line.
    """
    fwd_positions = sorted(
        df.loc[df["strand"] == "+", "pam_position"].tolist()
    )
    highlighted = []
    i = 0
    pam_set = set(fwd_positions)
    while i < len(sequence):
        if i in pam_set:
            highlighted.append(f"{_COLOR_PAM}{sequence[i:i + PAM_LENGTH]}{_COLOR_RESET}")
            i += PAM_LENGTH
        else:
            highlighted.append(sequence[i])
            i += 1
    return "".join(highlighted)


def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save the guide RNA candidate DataFrame to CSV."""
    df.to_csv(output_path, index=False)


def print_summary(sequence: str, df: pd.DataFrame) -> None:
    """Print DNA length, PAM count, PAM positions, and guide sequences."""
    print(f"DNA length: {len(sequence)} bp")
    print(f"Number of PAM sites: {len(df)}")
    print(f"PAM positions: {df['pam_position'].tolist()}")
    print("Guide RNA sequences:")
    for _, row in df.iterrows():
        print(f"  [{row['strand']}] pos={row['pam_position']:<6} "
              f"guide={row['guide_rna']}  PAM={row['pam_sequence']}")


def run_pipeline(fasta_path: str, output_csv: str = "guide_candidates.csv") -> pd.DataFrame:
    """End-to-end pipeline: read FASTA -> find PAMs -> generate guides -> save CSV."""
    sequence = read_fasta_sequence(fasta_path)
    df = build_guide_dataframe(sequence)

    print_summary(sequence, df)
    print("\nPAM-highlighted sequence (forward strand):")
    print(highlight_pam_in_sequence(sequence, df))

    save_to_csv(df, output_csv)
    print(f"\nSaved {len(df)} guide RNA candidates to: {output_csv}")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pam_guide_module.py <input.fasta> [output.csv]")
        sys.exit(1)

    fasta_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "guide_candidates.csv"
    run_pipeline(fasta_file, csv_file)