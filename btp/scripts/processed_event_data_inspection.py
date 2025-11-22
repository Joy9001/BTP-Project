from pathlib import Path

import numpy as np

# Replace with the path to one of your PROCESSED output files
processed_file_path = (
    Path.cwd()
    / "data"
    / "processed"
    / "Processed_Lowlight_event"
    / "00001"
    / "00001.npy"
)

try:
    # Load the processed tensor
    tensor = np.load(processed_file_path)

    print(f"--- Processed File: {processed_file_path} ---")
    print(f"Shape: {tensor.shape}")
    # Expected: (T, 2, 260, 346)
    # T should be approx Total_Duration / 50ms

    print(f"Data Type: {tensor.dtype}")

    # Check for "Silent Failure" (All Zeros)
    total_elements = tensor.size
    non_zero_elements = np.count_nonzero(tensor)
    sparsity = 1.0 - (non_zero_elements / total_elements)

    print("\n--- Content Health ---")
    print(f"Non-zero elements: {non_zero_elements}")
    print(f"Sparsity: {sparsity:.4%} (Should be high, e.g., >90%, but not 100%)")
    print(f"Max Value in Tensor: {tensor.max()}")

    if non_zero_elements == 0:
        print("\n[WARNING] Tensor is EMPTY! Denoising might be too aggressive.")
    else:
        print("\n[SUCCESS] Tensor contains data. Ready for Feature Extraction.")

except Exception as e:
    print(f"Error reading file: {e}")
