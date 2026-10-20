"""
deep_spcas9.py

Standalone prediction module for DeepSpCas9 (Kim et al., Science Advances 2019).
Computes predicted on-target activity (indel frequency score) for CRISPR/SpCas9 guide RNAs.

Published Architecture:
  - Input: 30-bp sequence context [4 bp upstream + 20 bp guide + 3 bp PAM (NGG) + 3 bp downstream]
  - 3 parallel Conv2d branches with kernel sizes (1, 3), (1, 5), (1, 7) and filter counts 100, 70, 40
  - ReLU activation
  - Average pooling with kernel (1, 2) and stride (1, 2)
  - Flattening and concatenation (matching TensorFlow channel-last format)
  - FC1: Linear(2790, 80) -> ReLU
  - FC2: Linear(80, 60) -> ReLU
  - Output: Linear(60, 1) -> continuous predicted activity score
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio.Seq import Seq

log = logging.getLogger("deep_spcas9")

CONTEXT_LENGTH = 30
UPSTREAM_FLANK = 4
GUIDE_LENGTH = 20
PAM_LENGTH = 3
DOWNSTREAM_FLANK = 3

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "deep_spcas9" / "DeepCas9_weights.pt"


class DeepSpCas9Model(nn.Module):
    """
    PyTorch implementation of the published DeepSpCas9 architecture (Kim et al., 2019).
    Faithfully reproduces layer dimensions, activations, pooling, and concatenation logic.

    Output Semantics & Unconstrained Linear Regression Head:
      The architecture terminates in a single linear regression head (`nn.Linear(60, 1)`)
      with no activation function, bounding, or clamping (no sigmoid or ReLU). The model
      was trained using MSE loss on experimentally measured indel frequencies (percentage
      scale, roughly 0% to 100%).

      For sequences with unfavorable nucleotide context or negligible cleavage activity,
      the unconstrained linear combination can evaluate to small negative values (e.g.
      -15.0 to 0.0). These negative values are authentic model predictions representing
      sub-detection / inactive targets, not an implementation or transformation error.
      Predictions are intentionally unclipped to maintain ranking fidelity and exact
      numerical parity with the published TensorFlow checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        # Parallel convolutional branches
        # PyTorch Conv2d input: [N, C=4, H=1, W=30]
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=100, kernel_size=(1, 3), padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=70, kernel_size=(1, 5), padding=0)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=40, kernel_size=(1, 7), padding=0)

        # Average pooling with stride 2
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)

        # Fully connected layers
        # Branch 1 out width: 14 -> 14 * 100 = 1400
        # Branch 2 out width: 13 -> 13 * 70  = 910
        # Branch 3 out width: 12 -> 12 * 40  = 480
        # Total concatenated features = 2790
        self.fc1 = nn.Linear(2790, 80)
        self.fc2 = nn.Linear(80, 60)
        self.out = nn.Linear(60, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape [N, 4, 1, 30] (one-hot encoded sequences)
        Returns:
            Tensor of shape [N, 1] with raw predicted activity scores
        """
        # Branch 1
        x1 = self.relu(self.conv1(x))
        x1 = self.pool(x1)
        x1 = x1.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # Branch 2
        x2 = self.relu(self.conv2(x))
        x2 = self.pool(x2)
        x2 = x2.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # Branch 3
        x3 = self.relu(self.conv3(x))
        x3 = self.pool(x3)
        x3 = x3.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # Concatenate features
        concat = torch.cat([x1, x2, x3], dim=1)

        # Fully connected layers
        h1 = self.relu(self.fc1(concat))
        h2 = self.relu(self.fc2(h1))
        out = self.out(h2)
        return out


def encode_sequences(sequences: list[str]) -> torch.Tensor:
    """
    One-hot encode a list of 30-bp DNA sequences for DeepSpCas9.
    
    Channels:
      0 -> A
      1 -> C
      2 -> G
      3 -> T

    Returns:
        torch.Tensor of shape [len(sequences), 4, 1, 30], dtype=torch.float32
    """
    n = len(sequences)
    arr = np.zeros((n, 4, 1, CONTEXT_LENGTH), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if len(seq) != CONTEXT_LENGTH:
            raise ValueError(f"Sequence '{seq}' has length {len(seq)}, expected {CONTEXT_LENGTH} bp")
        seq_upper = seq.upper()
        for pos, base in enumerate(seq_upper):
            if base == 'A':
                arr[i, 0, 0, pos] = 1.0
            elif base == 'C':
                arr[i, 1, 0, pos] = 1.0
            elif base == 'G':
                arr[i, 2, 0, pos] = 1.0
            elif base == 'T':
                arr[i, 3, 0, pos] = 1.0
            else:
                raise ValueError(f"Invalid base '{base}' at pos {pos} in sequence: {seq}")
    return torch.from_numpy(arr)


def extract_30bp_context(
    sequence: str,
    pam_position: int,
    strand: str,
) -> Optional[str]:
    """
    Extract the 30-bp context window required by DeepSpCas9 for a given guide candidate.
    
    Window layout (5' -> 3' relative to guide target strand):
      [4 bp upstream] + [20 bp guide] + [3 bp PAM] + [3 bp downstream]
      Total: 30 bp

    Args:
        sequence: Full reference DNA sequence (forward / + strand).
        pam_position: 0-based PAM coordinate mapped to original (+) strand.
        strand: '+' for forward strand, '-' for reverse strand.

    Returns:
        30-bp string if sufficient flanking context exists, otherwise None.
    """
    seq_len = len(sequence)
    seq_upper = sequence.upper()

    if strand == "+":
        # On forward strand:
        # Protospacer: sequence[pam_position - 20 : pam_position]
        # PAM: sequence[pam_position : pam_position + 3]
        # Upstream flank: sequence[pam_position - 24 : pam_position - 20]
        # Downstream flank: sequence[pam_position + 3 : pam_position + 6]
        start = pam_position - UPSTREAM_FLANK - GUIDE_LENGTH  # pam_position - 24
        end = pam_position + PAM_LENGTH + DOWNSTREAM_FLANK     # pam_position + 6

        if start < 0 or end > seq_len:
            return None
        return seq_upper[start:end]

    elif strand == "-":
        # On reverse strand:
        # Candidate generation was performed on rev_comp sequence.
        # pam_start_on_rc = seq_len - pam_position - PAM_LENGTH
        # Protospacer on rc: rev_comp[pam_start_rc - 20 : pam_start_rc]
        # Upstream flank on rc: rev_comp[pam_start_rc - 24 : pam_start_rc - 20]
        # Downstream flank on rc: rev_comp[pam_start_rc + 3 : pam_start_rc + 6]
        # Context on rc: rev_comp[pam_start_rc - 24 : pam_start_rc + 6]
        #
        # In terms of original (+) strand sequence coordinates:
        # rev_comp[pam_start_rc - 24 : pam_start_rc + 6] corresponds to
        # sequence[pam_position - 3 : pam_position + 27] reverse-complemented!
        start_fwd = pam_position - DOWNSTREAM_FLANK           # pam_position - 3
        end_fwd = pam_position + PAM_LENGTH + GUIDE_LENGTH + UPSTREAM_FLANK  # pam_position + 27

        if start_fwd < 0 or end_fwd > seq_len:
            return None

        genomic_window = seq_upper[start_fwd:end_fwd]
        return str(Seq(genomic_window).reverse_complement())

    else:
        raise ValueError(f"Unknown strand: '{strand}', expected '+' or '-'")


class DeepSpCas9Predictor:
    """
    High-level inference predictor for DeepSpCas9 on-target activity.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"DeepSpCas9 weights file not found: {self.weights_path}. "
                f"Please run convert_weights.py first."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = DeepSpCas9Model()
        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        log.info("Loaded DeepSpCas9 model from %s on %s", self.weights_path, self.device)

    def predict_scores(
        self,
        sequences: list[str],
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Run DeepSpCas9 inference on a list of 30-bp target context sequences.

        Args:
            sequences: List of 30-bp strings (4bp + 20bp guide + 3bp PAM + 3bp).
            batch_size: Batch size for inference.

        Returns:
            1D numpy array of float32 predictions (same length as sequences).
            Raw scores represent unconstrained linear predictions of indel percentage
            (Kim et al., 2019) and may produce negative values (e.g. down to -15) for
            guides with negligible/undetectable cleavage propensity. Raw scores are
            intentionally preserved without clipping.
        """
        if not sequences:
            return np.empty(0, dtype=np.float32)

        tensor_x = encode_sequences(sequences)
        n_samples = len(sequences)
        predictions = np.zeros(n_samples, dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_tensor = tensor_x[i : i + batch_size].to(self.device)
                batch_out = self.model(batch_tensor).squeeze(-1)
                predictions[i : i + batch_size] = batch_out.cpu().numpy()

        return predictions

    def score_candidates_dataframe(
        self,
        df: pd.DataFrame,
        fasta_sequence: str,
        batch_size: int = 512,
    ) -> tuple[pd.Series, dict]:
        """
        Extract 30-bp context for all candidate gRNAs in DataFrame and predict DeepSpCas9 scores.

        Args:
            df: DataFrame containing 'pam_position', 'strand', 'guide_rna' columns.
            fasta_sequence: Full reference DNA sequence (string).
            batch_size: Inference batch size.

        Returns:
            Tuple of (scores_series, summary_stats_dict)
        """
        n_candidates = len(df)
        contexts: list[Optional[str]] = []
        valid_indices: list[int] = []
        valid_contexts: list[str] = []

        for idx, row in df.iterrows():
            pos = int(row["pam_position"])
            strand = str(row["strand"])
            ctx = extract_30bp_context(fasta_sequence, pos, strand)
            contexts.append(ctx)
            if ctx is not None:
                valid_indices.append(idx)
                valid_contexts.append(ctx)

        scores = np.full(n_candidates, np.nan, dtype=np.float64)

        if valid_contexts:
            valid_preds = self.predict_scores(valid_contexts, batch_size=batch_size)
            for idx, pred in zip(valid_indices, valid_preds):
                scores[idx] = float(pred)

        scores_series = pd.Series(scores, index=df.index, name="deep_spcas9_score")

        n_predicted = len(valid_contexts)
        n_failed = n_candidates - n_predicted

        stats: dict = {
            "total_candidates": n_candidates,
            "successfully_predicted": n_predicted,
            "failed_edge_cases": n_failed,
        }

        if n_predicted > 0:
            valid_s = scores_series.dropna()
            stats.update({
                "min": round(float(valid_s.min()), 4),
                "max": round(float(valid_s.max()), 4),
                "mean": round(float(valid_s.mean()), 4),
                "median": round(float(valid_s.median()), 4),
                "std": round(float(valid_s.std()), 4) if len(valid_s) > 1 else 0.0,
            })

        return scores_series, stats
