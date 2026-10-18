"""
convert_weights.py

Phase 2 & Phase 3 validation:
Converts TF1 checkpoint weights to PyTorch state_dict and validates
the PyTorch DeepSpCas9 model against the TF1 reference outputs.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'Paired-Library', 'DeepCas9', 'DeepCas9_Final'
)
CHECKPOINT_PREFIX = 'PreTrain-Final-False-3-5-7-100-70-40-0.001-550-True-80-60'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

OUTPUT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'Paired-Library', 'DeepCas9', 'DeepCas9_weights.pt'
)

# Reference sequences from sample.txt
REF_SEQUENCES = [
    "TAAGAGAGTGGTAATAGAAGTGCCAGGTAT",
    "CCCTCATGGTGCAGCTAAAGGCCCAGGAGC",
]
REF_EXPECTED = [67.5565185546875, 56.930904388427734]
LENGTH = 30


class DeepSpCas9Model(nn.Module):
    """
    PyTorch re-implementation of the published DeepSpCas9 model (Kim et al., 2019).
    
    Architecture:
      - 3 parallel Conv2d branches with kernel sizes (1, 3), (1, 5), (1, 7)
        and filter counts 100, 70, 40
      - ReLU activation
      - Average pooling with kernel (1, 2), stride (1, 2)
      - Concatenation of flattened feature maps (matching TF channel-last layout)
      - FC1: Linear(2790, 80) -> ReLU
      - FC2: Linear(80, 60) -> ReLU
      - Output: Linear(60, 1)
    """
    def __init__(self):
        super().__init__()
        # Branch 1: kernel size (1, 3), 100 filters
        # TF input: [N, 1, 30, 4] -> PyTorch: [N, 4, 1, 30]
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=100, kernel_size=(1, 3), padding=0)
        # Branch 2: kernel size (1, 5), 70 filters
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=70, kernel_size=(1, 5), padding=0)
        # Branch 3: kernel size (1, 7), 40 filters
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=40, kernel_size=(1, 7), padding=0)

        # Average pooling with stride 2 and padding='same'
        # In TF: ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME'
        # On width dim:
        # conv1 out: 30 - 3 + 1 = 28 -> avg_pool(SAME, k=2, s=2) -> ceil(28/2) = 14
        # conv2 out: 30 - 5 + 1 = 26 -> avg_pool(SAME, k=2, s=2) -> ceil(26/2) = 13
        # conv3 out: 30 - 7 + 1 = 24 -> avg_pool(SAME, k=2, s=2) -> ceil(24/2) = 12
        # PyTorch AvgPool2d((1, 2), stride=(1, 2), padding=0) on even lengths 28, 26, 24 produces 14, 13, 12!
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)

        # Fully connected layers
        # node_num_0 = 14 * 100 = 1400
        # node_num_1 = 13 * 70  = 910
        # node_num_2 = 12 * 40  = 480
        # total = 2790
        self.fc1 = nn.Linear(2790, 80)
        self.fc2 = nn.Linear(80, 60)
        self.out = nn.Linear(60, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [N, 4, 1, 30]
        # Branch 1
        x1 = self.relu(self.conv1(x))    # [N, 100, 1, 28]
        x1 = self.pool(x1)               # [N, 100, 1, 14]
        # To match TF flattening [N, 1, 14, 100] -> [N, 1400]
        x1 = x1.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # Branch 2
        x2 = self.relu(self.conv2(x))    # [N, 70, 1, 26]
        x2 = self.pool(x2)               # [N, 70, 1, 13]
        x2 = x2.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # Branch 3
        x3 = self.relu(self.conv3(x))    # [N, 40, 1, 24]
        x3 = self.pool(x3)               # [N, 40, 1, 12]
        x3 = x3.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # Concatenate: [N, 2790]
        concat = torch.cat([x1, x2, x3], dim=1)

        # FC layers
        out = self.relu(self.fc1(concat))
        out = self.relu(self.fc2(out))
        out = self.out(out)  # [N, 1]
        return out


def convert_tf_to_pytorch():
    """Read TF1 checkpoint and map to PyTorch state_dict."""
    print("=" * 72)
    print("TF1 TO PYTORCH WEIGHT CONVERSION")
    print("=" * 72)
    print(f"Reading checkpoint: {CHECKPOINT_PATH}")

    reader = tf.train.load_checkpoint(CHECKPOINT_PATH)
    
    # Required variable map
    var_map = {
        "conv1_W": "conv1.weight",
        "conv1_b": "conv1.bias",
        "conv2_W": "conv2.weight",
        "conv2_b": "conv2.bias",
        "conv3_W": "conv3.weight",
        "conv3_b": "conv3.bias",
        "Fully_Connected_Layer1/W_fcl1": "fc1.weight",
        "Fully_Connected_Layer1/B_fcl1": "fc1.bias",
        "Fully_Connected_Layer2/W_fcl2": "fc2.weight",
        "Fully_Connected_Layer2/B_fcl2": "fc2.bias",
        "Output_Layer/W_out": "out.weight",
        "Output_Layer/B_out": "out.bias",
    }

    state_dict = {}

    for tf_name, pt_name in var_map.items():
        if not reader.has_tensor(tf_name):
            raise KeyError(f"CRITICAL ERROR: Required variable '{tf_name}' not found in TF1 checkpoint!")
        
        arr = reader.get_tensor(tf_name)
        print(f"Mapping '{tf_name}' (shape {arr.shape}) -> '{pt_name}'")

        if "conv" in tf_name and "_W" in tf_name:
            # TF conv weight shape: [H=1, W=filter_size, in_channels=4, out_channels=filter_num]
            # PyTorch Conv2d weight shape: [out_channels, in_channels, H, W]
            # Transpose: (3, 2, 0, 1)
            arr_pt = np.transpose(arr, (3, 2, 0, 1))
            tensor = torch.from_numpy(arr_pt.copy()).float()
        elif "W_fc" in tf_name or "W_out" in tf_name:
            # TF dense weight shape: [in_features, out_features]
            # PyTorch Linear weight shape: [out_features, in_features]
            arr_pt = np.transpose(arr, (1, 0))
            tensor = torch.from_numpy(arr_pt.copy()).float()
        else:
            # Biases: shape matches directly
            tensor = torch.from_numpy(arr.copy()).float()

        state_dict[pt_name] = tensor

    # Instantiate model and load state_dict with strict check
    model = DeepSpCas9Model()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Save converted weights
    os.makedirs(os.path.dirname(OUTPUT_WEIGHTS_PATH), exist_ok=True)
    torch.save(state_dict, OUTPUT_WEIGHTS_PATH)
    print(f"\nSuccessfully converted all weights and saved to: {OUTPUT_WEIGHTS_PATH}")
    print(f"Weight file size: {os.path.getsize(OUTPUT_WEIGHTS_PATH)} bytes")
    
    return model


def preprocess_for_pytorch(sequences):
    """
    Encode sequences for PyTorch model.
    Returns tensor of shape [N, 4, 1, 30]
    Encoding: A->channel 0, C->channel 1, G->channel 2, T->channel 3
    """
    N = len(sequences)
    arr = np.zeros((N, 4, 1, LENGTH), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for pos in range(LENGTH):
            base = seq[pos]
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


def validate_pytorch_model(model):
    """Compare PyTorch outputs on reference sequences against expected values."""
    print()
    print("=" * 72)
    print("PYTORCH MODEL VALIDATION ON REFERENCE SEQUENCES")
    print("=" * 72)

    x = preprocess_for_pytorch(REF_SEQUENCES)
    with torch.no_grad():
        preds = model(x).squeeze().numpy()

    print(f"{'Sequence':<35} {'PyTorch Output':>15} {'Expected':>15} {'Abs Diff':>12} {'Rel Diff (%)':>14}")
    print("-" * 95)

    max_abs_err = 0.0
    for i, (seq, pred) in enumerate(zip(REF_SEQUENCES, preds)):
        expected = REF_EXPECTED[i]
        abs_diff = abs(float(pred) - expected)
        rel_diff = (abs_diff / expected) * 100.0
        max_abs_err = max(max_abs_err, abs_diff)
        print(f"{seq:<35} {float(pred):>15.6f} {expected:>15.6f} {abs_diff:>12.8f} {rel_diff:>13.6f}%")

    print("-" * 95)
    print(f"Maximum Absolute Difference: {max_abs_err:.8f}")

    if max_abs_err < 1e-4:
        print(">>> VALIDATION PASSED: PyTorch output matches published reference values within numerical precision! <<<")
        return True
    else:
        print(">>> VALIDATION FAILED: Difference exceeds acceptable tolerance! <<<")
        return False


def main():
    model = convert_tf_to_pytorch()
    success = validate_pytorch_model(model)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
