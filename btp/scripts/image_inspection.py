import os
from pathlib import Path

import numpy as np
from PIL import Image

# Replace with the path to ONE of your raw image files
# image_path = (
#     Path.cwd() / "data" / "raw" / "Dataset" / "Lowlight_Images" / "00001" / "00001.png"
# )

image_path = (
    Path.cwd()
    / "data"
    / "raw"
    / "Dataset"
    / "Normalight_Images"
    / "00001"
    / "00001.png"
)

try:
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
    else:
        with Image.open(image_path) as img:
            print(f"--- Image File: {image_path} ---")
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode} (RGB, L, etc.)")
            print(f"Size (WxH): {img.size}")

            # Convert to numpy to check values
            img_np = np.array(img)
            print(f"Shape: {img_np.shape}")
            print(f"Data Type: {img_np.dtype}")
            print(f"Min Value: {img_np.min()}")
            print(f"Max Value: {img_np.max()}")

            # Check Aspect Ratio
            ar = img.size[0] / img.size[1]
            print(f"Aspect Ratio: {ar:.2f}")

            # Compare with Event Sensor
            # DAVIS346 Aspect Ratio = 346 / 260 = 1.33
            print("Event Sensor AR: 1.33")
            if abs(ar - 1.33) > 0.1:
                print(
                    "⚠️ Mismatch: Image AR differs significantly from Event AR. Cropping might be needed."
                )
            else:
                print("✅ Match: Aspect Ratios are similar.")

except Exception as e:
    print(f"Error inspecting image: {e}")
