"""
inspect_checkpoint.py

Phase 1 validation script: Inspect the DeepSpCas9 TF1 checkpoint
to extract exact variable names, tensor shapes, and run reference inference.

This script is run ONCE for validation purposes only.
"""

import os
import sys
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warnings

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Paths
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'Paired-Library', 'DeepCas9', 'DeepCas9_Final'
)
CHECKPOINT_PREFIX = 'PreTrain-Final-False-3-5-7-100-70-40-0.001-550-True-80-60'
SAMPLE_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'Paired-Library', 'DeepCas9', 'dataset', 'sample.txt'
)

# Reference sequences from sample.txt
REF_SEQUENCES = [
    "TAAGAGAGTGGTAATAGAAGTGCCAGGTAT",
    "CCCTCATGGTGCAGCTAAAGGCCCAGGAGC",
]
# Expected outputs from README: 67.5565185546875, 56.930904388427734
REF_EXPECTED = [67.5565185546875, 56.930904388427734]

LENGTH = 30


def inspect_checkpoint_variables():
    """List all variables in the checkpoint with names, shapes, dtypes."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
    
    print("=" * 72)
    print("CHECKPOINT INSPECTION")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print()
    
    # Use tf.train.list_variables (works in TF2 compat mode)
    variables = tf.train.list_variables(checkpoint_path)
    
    print(f"Total variables: {len(variables)}")
    print()
    print(f"{'Variable Name':<55} {'Shape':<25} {'Size':>10}")
    print("-" * 92)
    
    var_info = []
    for name, shape in variables:
        size = 1
        for s in shape:
            size *= s
        print(f"{name:<55} {str(shape):<25} {size:>10}")
        var_info.append({
            "name": name,
            "shape": list(shape),
            "size": size,
        })
    
    print()
    print("=" * 72)
    
    # Now load actual values to verify statistics
    print()
    print("VARIABLE STATISTICS")
    print("-" * 72)
    
    reader = tf.train.load_checkpoint(checkpoint_path)
    for name, shape in variables:
        tensor = reader.get_tensor(name)
        print(f"{name}: dtype={tensor.dtype}, min={tensor.min():.6f}, max={tensor.max():.6f}, "
              f"mean={tensor.mean():.6f}, std={tensor.std():.6f}")
        var_info_entry = next(v for v in var_info if v["name"] == name)
        var_info_entry["dtype"] = str(tensor.dtype)
        var_info_entry["min"] = float(tensor.min())
        var_info_entry["max"] = float(tensor.max())
        var_info_entry["mean"] = float(tensor.mean())
        var_info_entry["std"] = float(tensor.std())
    
    return var_info


def preprocess_seq(sequences):
    """One-hot encode sequences exactly as in published code."""
    DATA_X = np.zeros((len(sequences), 1, LENGTH, 4), dtype=np.float32)
    for l in range(len(sequences)):
        for i in range(LENGTH):
            ch = sequences[l][i].upper()
            if ch == 'A':
                DATA_X[l, 0, i, 0] = 1
            elif ch == 'C':
                DATA_X[l, 0, i, 1] = 1
            elif ch == 'G':
                DATA_X[l, 0, i, 2] = 1
            elif ch == 'T':
                DATA_X[l, 0, i, 3] = 1
            else:
                raise ValueError(f"Non-ATGC character '{ch}' at position {i} in sequence {l}")
    return DATA_X


def run_tf1_reference_inference():
    """Run the original TF1 model on reference sequences to get ground-truth outputs."""
    print()
    print("=" * 72)
    print("TF1 REFERENCE INFERENCE")
    print("=" * 72)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
    
    # Hyperparameters from checkpoint filename
    filter_size = [3, 5, 7]
    filter_num = [100, 70, 40]
    node_1 = 80
    node_2 = 60
    
    # Preprocess
    TEST_X = preprocess_seq(REF_SEQUENCES)
    print(f"Input shape: {TEST_X.shape}")
    
    # Build model using TF1 compat
    tf.compat.v1.reset_default_graph()
    
    inputs = tf.compat.v1.placeholder(tf.float32, [None, 1, LENGTH, 4])
    is_training = tf.compat.v1.placeholder(tf.bool)
    
    def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
        weights = tf.compat.v1.Variable(
            tf.compat.v1.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W'
        )
        bias = tf.compat.v1.Variable(
            tf.compat.v1.truncated_normal([num_filters]), name=name + '_b'
        )
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='VALID')
        out_layer += bias
        out_layer = tf.nn.relu(out_layer)
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 1, 2, 1]
        out_layer = tf.nn.avg_pool2d(out_layer, ksize=ksize, strides=strides, padding='SAME')
        return out_layer
    
    L_pool_0 = create_new_conv_layer(inputs, 4, filter_num[0], [1, filter_size[0]], [1, 2], name='conv1')
    L_pool_1 = create_new_conv_layer(inputs, 4, filter_num[1], [1, filter_size[1]], [1, 2], name='conv2')
    L_pool_2 = create_new_conv_layer(inputs, 4, filter_num[2], [1, filter_size[2]], [1, 2], name='conv3')
    
    with tf.compat.v1.variable_scope('Fully_Connected_Layer1'):
        layer_node_0 = int((LENGTH - filter_size[0]) / 2) + 1
        node_num_0 = layer_node_0 * filter_num[0]
        layer_node_1 = int((LENGTH - filter_size[1]) / 2) + 1
        node_num_1 = layer_node_1 * filter_num[1]
        layer_node_2 = int((LENGTH - filter_size[2]) / 2) + 1
        node_num_2 = layer_node_2 * filter_num[2]
        
        L_flatten_0 = tf.reshape(L_pool_0, [-1, node_num_0])
        L_flatten_1 = tf.reshape(L_pool_1, [-1, node_num_1])
        L_flatten_2 = tf.reshape(L_pool_2, [-1, node_num_2])
        L_flatten = tf.concat([L_flatten_0, L_flatten_1, L_flatten_2], 1, name='concat')
        node_num = node_num_0 + node_num_1 + node_num_2
        
        W_fcl1 = tf.compat.v1.get_variable("W_fcl1", shape=[node_num, node_1])
        B_fcl1 = tf.compat.v1.get_variable("B_fcl1", shape=[node_1])
        L_fcl1_pre = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
        L_fcl1 = tf.nn.relu(L_fcl1_pre)
    
    with tf.compat.v1.variable_scope('Fully_Connected_Layer2'):
        W_fcl2 = tf.compat.v1.get_variable("W_fcl2", shape=[node_1, node_2])
        B_fcl2 = tf.compat.v1.get_variable("B_fcl2", shape=[node_2])
        L_fcl2_pre = tf.nn.bias_add(tf.matmul(L_fcl1, W_fcl2), B_fcl2)
        L_fcl2 = tf.nn.relu(L_fcl2_pre)
    
    with tf.compat.v1.variable_scope('Output_Layer'):
        W_out = tf.compat.v1.get_variable("W_out", shape=[node_2, 1])
        B_out = tf.compat.v1.get_variable("B_out", shape=[1])
        outputs = tf.nn.bias_add(tf.matmul(L_fcl2, W_out), B_out)
    
    # Run inference
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint_path)
        
        predictions = sess.run(outputs, feed_dict={inputs: TEST_X, is_training: False})
    
    print()
    print("Reference Inference Results:")
    print(f"{'Sequence':<35} {'TF1 Output':>12} {'Expected':>12} {'Diff':>10}")
    print("-" * 72)
    
    results = []
    for i, (seq, pred) in enumerate(zip(REF_SEQUENCES, predictions)):
        pred_val = float(pred[0])
        expected = REF_EXPECTED[i]
        diff = abs(pred_val - expected)
        print(f"{seq:<35} {pred_val:>12.6f} {expected:>12.6f} {diff:>10.6f}")
        results.append({
            "sequence": seq,
            "tf1_output": pred_val,
            "expected": expected,
            "abs_diff": diff,
        })
    
    print()
    return results


def main():
    output = {}
    
    # Step 1: Inspect variables
    var_info = inspect_checkpoint_variables()
    output["variables"] = var_info
    
    # Step 2: Run reference inference
    ref_results = run_tf1_reference_inference()
    output["reference_inference"] = ref_results
    
    # Save results
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'Paired-Library', 'DeepCas9',
        'checkpoint_inspection.json'
    )
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved inspection results to: {output_path}")


if __name__ == "__main__":
    main()
