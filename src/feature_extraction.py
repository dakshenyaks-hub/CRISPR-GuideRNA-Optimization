"""
feature_extraction.py

Module 3 of the Multi-Objective CRISPR Guide RNA Design Framework.

Consumes the guide candidates produced by Module 1 (PAM detection) and
computes a full feature vector for each guide RNA, spanning:

    A. Sequence features       - GC content, composition, length, motifs
    B. Thermodynamic features  - melting temperature, duplex stability (dG)
    C. Structural features     - MFE secondary structure, folding energy
    D. Specificity features    - Bowtie genome alignment, off-targets, mismatch score
    E. Efficiency features     - Doench 2016-style on-target efficiency proxy

Input : data/guide_candidates.csv   (from pam_guide_module.py)
Output: data/gRNA_features.csv

Author: Bioinformatics Development Team
"""

from __future__ import annotations

import argparse
import logging
import math
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import RNA  # ViennaRNA python bindings
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import gc_fraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_extraction")

PAM_LENGTH = 3
GUIDE_LENGTH = 20
NN_TABLE = mt.DNA_NN4          # SantaLucia & Hicks (2004) unified NN parameters
COMPLEMENT = str.maketrans("ACGT", "TGCA")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    input_csv: str = "data/guide_candidates.csv"
    output_csv: str = "data/gRNA_features.csv"
    genome_fasta: Optional[str] = None      # reference genome, for context + Bowtie index build
    bowtie_index: Optional[str] = None      # prebuilt Bowtie index prefix (skips build step)
    max_mismatches: int = 3                 # Bowtie -v mismatch tolerance
    threads: int = 4
    defer_bowtie_to_cfd: bool = False       # whether alignment is deferred to CFD stage


# --------------------------------------------------------------------------- #
# A. Sequence Features
# --------------------------------------------------------------------------- #
def compute_gc_content(seq: str) -> float:
    """Fraction of G/C bases in [0, 1]."""
    return round(gc_fraction(seq), 4)


def compute_nucleotide_counts(seq: str) -> dict:
    """Raw per-base counts."""
    return {f"{base}_count": seq.count(base) for base in "ACGT"}


def compute_nucleotide_composition(seq: str) -> dict:
    """Per-base fractional composition."""
    n = len(seq)
    return {f"frac_{base}": round(seq.count(base) / n, 4) for base in "ACGT"}


def longest_homopolymer_run(seq: str) -> int:
    """Length of the longest single-base run (e.g. 'TTTT' -> 4)."""
    longest = current = 1
    for i in range(1, len(seq)):
        current = current + 1 if seq[i] == seq[i - 1] else 1
        longest = max(longest, current)
    return longest


def detect_sequence_motifs(seq: str) -> dict:
    """Flag motifs known to affect Pol III expression / cloning."""
    return {
        "has_poly_t_terminator": "TTTT" in seq,       # Pol III termination signal
        "max_homopolymer_run": longest_homopolymer_run(seq),
        "starts_with_g": seq.startswith("G"),          # favorable for U6 promoter
    }


def extract_sequence_features(guide: str) -> dict:
    features = {"length": len(guide), "gc_content": compute_gc_content(guide)}
    features.update(compute_nucleotide_counts(guide))
    features.update(compute_nucleotide_composition(guide))
    features.update(detect_sequence_motifs(guide))
    return features


# --------------------------------------------------------------------------- #
# B. Thermodynamic Features
# --------------------------------------------------------------------------- #
def compute_melting_temperature(guide: str) -> float:
    """Nearest-neighbor melting temperature (deg C) of the guide:target duplex."""
    return round(mt.Tm_NN(guide, nn_table=NN_TABLE), 2)


def compute_duplex_stability(guide: str, temp_c: float = 37.0) -> float:
    """
    Gibbs free energy (kcal/mol) of guide:target duplex formation using
    unified SantaLucia & Hicks (2004) nearest-neighbor parameters.
    More negative -> more stable duplex.
    """
    seq = guide.upper()
    c_seq = seq.translate(COMPLEMENT)  # base-wise complement, same 5'->3' index order
    d_h = d_s = 0.0

    # Initiation terms (matches Biopython's Tm_NN initiation convention)
    d_h += NN_TABLE["init"][0]
    d_s += NN_TABLE["init"][1]
    for end_base in (seq[0], seq[-1]):
        key = "init_A/T" if end_base in "AT" else "init_G/C"
        d_h += NN_TABLE[key][0]
        d_s += NN_TABLE[key][1]

    # Nearest-neighbor stacking terms
    for i in range(len(seq) - 1):
        step = seq[i : i + 2] + "/" + c_seq[i : i + 2]
        if step in NN_TABLE:
            d_h_i, d_s_i = NN_TABLE[step]
        else:
            rev = step[::-1]
            d_h_i, d_s_i = NN_TABLE[rev]
        d_h += d_h_i
        d_s += d_s_i

    temp_k = temp_c + 273.15
    delta_g = d_h - (temp_k * d_s / 1000.0)  # kcal/mol
    return round(delta_g, 3)


def extract_thermodynamic_features(guide: str) -> dict:
    return {
        "melting_temp_c": compute_melting_temperature(guide),
        "delta_g_kcal_mol": compute_duplex_stability(guide),
    }


# --------------------------------------------------------------------------- #
# C. Structural Features
# --------------------------------------------------------------------------- #
def fold_secondary_structure(guide: str) -> tuple[str, float]:
    """Minimum free energy (MFE) secondary structure via ViennaRNA."""
    rna_seq = guide.upper().replace("T", "U")
    structure, mfe = RNA.fold(rna_seq)
    return structure, mfe


def extract_structural_features(guide: str) -> dict:
    structure, mfe = fold_secondary_structure(guide)
    paired_bases = structure.count("(") + structure.count(")")
    return {
        "secondary_structure": structure,
        "folding_energy_kcal_mol": round(mfe, 2),
        "paired_fraction": round(paired_bases / len(structure), 4),
    }


# --------------------------------------------------------------------------- #
# D. Specificity Features (Bowtie genome alignment)
# --------------------------------------------------------------------------- #
def _running_on_windows() -> bool:
    """True when this Python process itself is running under Windows."""
    return platform.system() == "Windows"


def _windows_path_to_wsl(path: str) -> str:
    """
    Convert a Windows path (e.g. 'data\\hg38_index\\hg38_index' or
    'C:\\Users\\me\\data\\hg38_index') to its WSL mount-point equivalent
    (e.g. '/mnt/c/Users/me/data/hg38_index'). Relative paths are resolved
    against the current working directory first, since WSL needs an
    absolute path to locate the file under /mnt/<drive>/...
    """
    resolved = Path(path).resolve()
    drive_letter = resolved.drive.rstrip(":").lower()
    tail = str(resolved)[len(resolved.drive):].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive_letter}/{tail}"


def _bowtie_command(args: list[str]) -> list[str]:
    """
    Build the OS-appropriate command to invoke Bowtie.
    On Windows, Bowtie is not installed natively (WinError 2) - it lives
    inside WSL Ubuntu, so the call is routed through `wsl <args>`.
    On Linux/macOS, Bowtie is called directly as before.
    """
    return ["wsl"] + args if _running_on_windows() else args


def _bowtie_path(path: str) -> str:
    """Translate a filesystem path for Bowtie's consumption, based on OS."""
    return _windows_path_to_wsl(path) if _running_on_windows() else path


def build_bowtie_index(genome_fasta: str, index_prefix: str) -> None:
    """Build a Bowtie1 index from a reference genome FASTA."""
    log.info("Building Bowtie index from %s ...", genome_fasta)
    cmd = _bowtie_command(
        ["bowtie-build", "-q", _bowtie_path(genome_fasta), _bowtie_path(index_prefix)]
    )
    subprocess.run(cmd, check=True, capture_output=True)


def align_guides_with_bowtie(
    guides: list[str], index_prefix: str, max_mismatches: int = 3, threads: int = 4
) -> dict[str, dict]:
    """
    Align every guide against the genome index with Bowtie (-v mode, report
    up to 1000 alignments per guide) and derive per-guide off-target count +
    mismatch score.

    mismatch_score: mean edit distance across off-target alignments
                    (higher = more divergent off-targets = safer guide).
                    NaN when a guide has no off-target hits.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        query_fasta = Path(tmpdir) / "guides.fa"
        with open(query_fasta, "w") as fh:
            for i, g in enumerate(guides):
                fh.write(f">guide_{i}\n{g}\n")

        # tempfile.TemporaryDirectory() on Windows returns a Windows path
        # (e.g. C:\Users\...\AppData\Local\Temp\...) - translate it too,
        # so WSL's bowtie can actually see the query FASTA.
        cmd = _bowtie_command([
            "bowtie",
            "-f",                       # query is FASTA
            "-v", str(max_mismatches),  # mismatches allowed (no indels)
            "-k", "1000",               # report up to 1000 alignments per guide
            "--best",
            "-p", str(threads),
            "-x", _bowtie_path(index_prefix),  # use -x option for index prefix
            _bowtie_path(str(query_fasta)),
        ])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode not in (0, 1):  # bowtie returns 1 if 0 reads aligned
            # Log detailed Bowtie execution info for debugging large runs
            log.error(f"Bowtie returncode: {result.returncode}")
            log.error(f"Bowtie stderr: {result.stderr}")
            tail_stdout = result.stdout[-1000:] if result.stdout else ""
            log.error(f"Bowtie stdout (last 1000 chars): {tail_stdout}")
            raise RuntimeError(f"Bowtie failed: {result.stderr}")

        # ------------------------------------------------------------------
        # Bowtie 1 default (non-SAM) output: exactly 8 tab-separated columns
        # (0-indexed):
        #
        #   [0] read name          [1] strand (+/-)
        #   [2] reference name     [3] 0-based offset
        #   [4] read sequence      [5] read qualities (all 'I' for FASTA input)
        #   [6] reserved           [7] mismatch descriptors
        #
        # Column [7] is a comma-separated list of "offset:ref>alt" tokens, one
        # per mismatch.  For a perfect (0-mismatch) alignment the field is an
        # empty string; in some Bowtie builds the trailing tab is omitted, so
        # the field may simply be absent.
        #
        # WSL → Windows subprocess pipe artefact: Bowtie (Linux binary) emits
        # LF-only lines, but the Windows pipe layer can leave a stray \r in the
        # captured text even when text=True is set.  splitlines() strips the
        # trailing \n but NOT a bare \r, so fields[7] for a 0-mismatch line
        # becomes "\r" instead of "" and the empty-check fails.  We therefore
        # call .strip() on edit_str before every comparison.
        # ------------------------------------------------------------------
        alignments: dict[str, list[int]] = {f"guide_{i}": [] for i in range(len(guides))}
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 5:
                continue                # skip blank or malformed lines
            read_id = fields[0]
            if read_id not in alignments:
                continue                # ignore stderr progress lines echoed to stdout

            # Safely read the mismatch descriptor field (index 7).
            # .strip() removes any stray \r or whitespace left by the WSL pipe.
            edit_str = fields[7].strip() if len(fields) > 7 else ""

            # Count mismatches: each comma-separated token is one mismatch.
            # An absent or empty field means 0 mismatches.
            n_mismatches = 0 if edit_str == "" else len(edit_str.split(","))
            alignments[read_id].append(n_mismatches)

    stats = {}
    for i, g in enumerate(guides):
        hits = alignments[f"guide_{i}"]

        # Tally alignments at each mismatch level (0 … max_mismatches).
        level_counts: dict[int, int] = {}
        for mm in hits:
            level_counts[mm] = level_counts.get(mm, 0) + 1

        # Deduct exactly ONE 0-mismatch hit (the expected on-target locus).
        # If the guide has multiple perfect-match sites, the extras are genuine
        # off-targets and must be counted.  Clamp to 0 for the edge case where
        # no 0-mismatch hit was found at all (unusual but possible).
        off_0mm = max(level_counts.get(0, 0) - 1, 0)
        off_1mm = level_counts.get(1, 0)
        off_2mm = level_counts.get(2, 0)
        off_3mm = level_counts.get(3, 0)

        off_target_count = off_0mm + off_1mm + off_2mm + off_3mm

        # Reconstruct a flat list of edit distances for all off-target hits so
        # that mismatch_score remains backward-compatible with existing callers.
        off_mm_values = (
            [0] * off_0mm
            + [1] * off_1mm
            + [2] * off_2mm
            + [3] * off_3mm
        )

        # Bowtie is run with -k 1000, which caps output at 1000 alignments
        # per guide.  After deducting one on-target hit the maximum observable
        # off-target count is 999.  When off_target_count reaches that ceiling
        # the true count is unknown (≥ 999), so we flag the guide.
        _K_LIMIT = 999  # -k 1000 minus the one on-target deduction
        stats[g] = {
            # Per-level off-target counts (0-mismatch bucket excludes the
            # single expected on-target hit).
            "off_targets_0mm": off_0mm,
            "off_targets_1mm": off_1mm,
            "off_targets_2mm": off_2mm,
            "off_targets_3mm": off_3mm,
            # Aggregate metrics preserved for backward compatibility.
            "off_target_count": off_target_count,
            "mismatch_score": (
                round(sum(off_mm_values) / len(off_mm_values), 3)
                if off_mm_values else float("nan")
            ),
            # True when the guide hit the -k 1000 reporting cap; the real
            # off-target count could be higher than off_target_count.
            "off_target_capped": off_target_count >= _K_LIMIT,
        }
    return stats


def extract_specificity_features(guides: list[str], config: FeatureConfig) -> dict[str, dict]:
    """Return {guide: {off_target_count, mismatch_score}} for all guides."""
    index_prefix = config.bowtie_index
    if index_prefix is None and config.genome_fasta is not None:
        index_prefix = str(Path(tempfile.mkdtemp()) / "genome_index")
        build_bowtie_index(config.genome_fasta, index_prefix)

    if index_prefix is None:
        if getattr(config, "defer_bowtie_to_cfd", False):
            log.info(
                "Bowtie alignment intentionally deferred to CFD stage "
                "for single-pass CFD scoring and genomic PAM extraction."
            )
        else:
            log.warning(
                "No reference genome or Bowtie index provided - "
                "specificity features will be reported as NaN."
            )
        return {
            g: {
                "off_target_count": float("nan"),
                "mismatch_score": float("nan"),
                "off_target_capped": False,
            }
            for g in guides
        }

    return align_guides_with_bowtie(guides, index_prefix, config.max_mismatches, config.threads)


# --------------------------------------------------------------------------- #
# E. Efficiency Features
# --------------------------------------------------------------------------- #
def doench2016_proxy_score(guide: str) -> float:
    """
    Simplified on-target efficiency proxy inspired by Doench et al. (2016)
    Rule Set 2 trends (position-specific nucleotide preference, GC content,
    homopolymer penalty). This is NOT the official gradient-boosted Rule Set 2
    model - it is a lightweight, dependency-free surrogate intended as a
    placeholder feature. For production scoring, replace this function with
    a call to the official Azimuth / Rule Set 2 model.

    Returns a score in [0, 1], higher = predicted more efficient.
    """
    seq = guide.upper()
    score = 0.5  # neutral baseline

    # GC content sweet spot ~40-60% (bell-shaped penalty outside range)
    gc = gc_fraction(seq)
    score += 0.15 * (1 - min(abs(gc - 0.5) / 0.5, 1.0))

    # Position -1 (adjacent to PAM) G/C is favorable
    if seq[-1] in "GC":
        score += 0.10

    # Position 1 (5' end, transcription start) G is favorable for U6
    if seq[0] == "G":
        score += 0.10

    # Penalize long homopolymer runs (reduces Pol III efficiency / risk of early termination)
    if longest_homopolymer_run(seq) >= 4:
        score -= 0.15

    return round(min(max(score, 0.0), 1.0), 4)


def extract_efficiency_features(guide: str) -> dict:
    return {"doench2016_proxy_score": doench2016_proxy_score(guide)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def extract_all_features(
    df: pd.DataFrame, config: FeatureConfig, gene_name: str = ""
) -> pd.DataFrame:
    """Compute A/B/C/E features per-guide, and D (specificity) in one batch call."""
    per_guide_records = []
    for guide in df["guide_rna"]:
        record = {}
        record.update(extract_sequence_features(guide))
        record.update(extract_thermodynamic_features(guide))
        record.update(extract_structural_features(guide))
        record.update(extract_efficiency_features(guide))
        per_guide_records.append(record)

    features_df = pd.DataFrame(per_guide_records)

    # --- Specificity features (Bowtie) ---
    specificity = extract_specificity_features(df["guide_rna"].tolist(), config)

    # Aggregate off-target metrics
    features_df["off_target_count"] = [specificity[g]["off_target_count"] for g in df["guide_rna"]]
    features_df["mismatch_score"] = [specificity[g]["mismatch_score"] for g in df["guide_rna"]]

    # Per-mismatch-level breakdown (already computed inside align_guides_with_bowtie)
    for mm_col in ("off_targets_0mm", "off_targets_1mm", "off_targets_2mm", "off_targets_3mm"):
        features_df[mm_col] = [
            specificity[g].get(mm_col, float("nan")) for g in df["guide_rna"]
        ]

    # off_target_capped is True when the guide reached the -k 1000 Bowtie
    # reporting limit (observable off-target count >= 999), meaning the real
    # count may be higher than off_target_count.
    features_df["off_target_capped"] = [specificity[g]["off_target_capped"] for g in df["guide_rna"]]

    # --- Combine candidate info + features ---
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # --- Add guide_id and gene columns for identification ---
    gene_label = gene_name.strip().upper() if gene_name else "UNKNOWN"
    result.insert(0, "gene", gene_label)
    result.insert(
        0,
        "guide_id",
        result.apply(
            lambda row: f"{gene_label}_{row['pam_position']}_{row['strand']}", axis=1
        ),
    )

    return result


def generate_quality_summary(df: pd.DataFrame) -> str:
    """
    Generate a data quality summary for the feature table.
    Returns a formatted multi-line string and also logs it.
    """
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("DATA QUALITY SUMMARY")
    lines.append("-" * 72)

    # Guide counts
    total = len(df)
    valid_20nt = int((df["guide_length"] == GUIDE_LENGTH).sum()) if "guide_length" in df.columns else total
    invalid = total - valid_20nt
    duplicates = int(df["guide_rna"].duplicated().sum()) if "guide_rna" in df.columns else 0

    lines.append(f"  Total candidates:      {total}")
    lines.append(f"  Valid 20-nt guides:    {valid_20nt}")
    lines.append(f"  Invalid guides:        {invalid}")
    lines.append(f"  Duplicate guides:      {duplicates}")

    # Strand distribution
    if "strand" in df.columns:
        plus = int((df["strand"] == "+").sum())
        minus = int((df["strand"] == "-").sum())
        lines.append(f"  + strand guides:       {plus}")
        lines.append(f"  - strand guides:       {minus}")

    # Feature statistics
    feature_stats = [
        ("gc_content", "GC content", ""),
        ("melting_temp_c", "Tm", "°C"),
        ("delta_g_kcal_mol", "ΔG (duplex)", "kcal/mol"),
        ("folding_energy_kcal_mol", "MFE (folding)", "kcal/mol"),
        ("doench2016_proxy_score", "Doench proxy", ""),
        ("paired_fraction", "Paired fraction", ""),
    ]

    lines.append("")
    lines.append("  Feature Statistics:")
    lines.append(f"  {'Feature':<25s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'NaN':>6s}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    for col, label, unit in feature_stats:
        if col not in df.columns:
            continue
        series = df[col]
        nan_count = int(series.isna().sum())
        if series.notna().any():
            suffix = f" {unit}" if unit else ""
            lines.append(
                f"  {label:<25s} {series.min():>10.4f} {series.max():>10.4f} "
                f"{series.mean():>10.4f} {nan_count:>6d}{suffix}"
            )
        else:
            lines.append(f"  {label:<25s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {nan_count:>6d}")

    # Off-target specificity
    if "off_target_count" in df.columns:
        ot_series = df["off_target_count"]
        ot_nan = int(ot_series.isna().sum())
        if ot_series.notna().any():
            lines.append(
                f"  {'Off-target count':<25s} {ot_series.min():>10.0f} {ot_series.max():>10.0f} "
                f"{ot_series.mean():>10.2f} {ot_nan:>6d}"
            )
        else:
            lines.append(
                f"  {'Off-target count':<25s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {ot_nan:>6d}"
            )

    # Missing value summary
    lines.append("")
    lines.append("  Missing Values per Feature:")
    nan_cols = df.isnull().sum()
    has_nans = False
    for col_name, nan_count in nan_cols.items():
        if nan_count > 0:
            lines.append(f"    {col_name}: {nan_count}")
            has_nans = True
    if not has_nans:
        lines.append("    (none)")

    lines.append("=" * 72)
    lines.append("")

    summary_text = "\n".join(lines)
    for line in lines:
        if line.strip():
            log.info(line)
    return summary_text


def run_pipeline(
    config: FeatureConfig, gene_name: str = ""
) -> pd.DataFrame:
    log.info("Loading guide candidates from %s", config.input_csv)
    candidates = pd.read_csv(config.input_csv)
    if candidates.empty:
        raise ValueError(f"No guide candidates found in {config.input_csv}")

    log.info("Extracting features for %d guides ...", len(candidates))
    result_df = extract_all_features(candidates, config, gene_name=gene_name)

    Path(config.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(config.output_csv, index=False)
    log.info("Saved feature table (%d rows x %d cols) to %s", *result_df.shape, config.output_csv)

    log.info(
        "Summary | mean GC: %.3f | mean Tm: %.2f C | mean dG: %.2f kcal/mol | mean MFE: %.2f kcal/mol",
        result_df["gc_content"].mean(),
        result_df["melting_temp_c"].mean(),
        result_df["delta_g_kcal_mol"].mean(),
        result_df["folding_energy_kcal_mol"].mean(),
    )

    # Print data quality summary
    generate_quality_summary(result_df)

    return result_df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> FeatureConfig:
    parser = argparse.ArgumentParser(description="CRISPR guide RNA feature extraction pipeline.")
    parser.add_argument("--input", default="data/guide_candidates.csv")
    parser.add_argument("--output", default="data/gRNA_features.csv")
    parser.add_argument("--genome-fasta", default=None, help="Reference genome FASTA (builds a Bowtie index).")
    parser.add_argument("--bowtie-index", default=None, help="Prebuilt Bowtie index prefix (skips index build).")
    parser.add_argument("--max-mismatches", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    return FeatureConfig(
        input_csv=args.input,
        output_csv=args.output,
        genome_fasta=args.genome_fasta,
        bowtie_index=args.bowtie_index,
        max_mismatches=args.max_mismatches,
        threads=args.threads,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())