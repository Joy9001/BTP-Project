from pathlib import Path

import numpy as np

# Replace with one of your PROCESSED image files
processed_img_path = (
    Path.cwd()
    / "data"
    / "processed"
    / "Processed_Lowlight_Images"
    / "00001"
    / "00001.npy"
)

try:
    img_tensor = np.load(processed_img_path)
    print(f"--- Processed Image: {processed_img_path} ---")
    print(f"Shape: {img_tensor.shape}")
    # EXPECTED: (3, 288, 352)

    print(f"Data Type: {img_tensor.dtype}")
    # EXPECTED: float32

    print(f"Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    # EXPECTED: [0.000, 1.000]

    if img_tensor.shape == (3, 288, 352):
        print("✅ Shape matches Event Branch requirements!")
    else:
        print("❌ Shape Mismatch! Event Branch uses padded (288, 352).")

except Exception as e:
    print(f"Error: {e}")
