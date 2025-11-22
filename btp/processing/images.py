from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class ImageProcessor:
    """Processing pipeline for Low-Light RGB Images.
    Key steps:
    1. CLAHE (Contrast Enhancement) to reveal objects in the dark.
    2. Padding to match the Event Tensor shape (multiples of 32).
    3. Saving as .npy for fast loading.
    """

    # Target Resolution (Must match Event Branch output)
    # 260 -> padded to 288
    # 346 -> padded to 352
    TARGET_SHAPE = (288, 352)

    def __init__(self):
        print("ImageProcessor initialized (CPU-based for OpenCV compatibility)")

    @staticmethod
    def enhance_image(image_bgr):
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to the L-channel of the LAB color space.
        This brightens the image without washing out colors.
        """
        # 1. Convert to LAB color space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 3. Merge and convert back to BGR
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        image_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Optional: Light Denoising (FastNlMeans is good but slow.
        # We use a slight Gaussian blur to reduce grain if needed,
        # but usually CNNs handle noise well if contrast is good).
        return image_enhanced

    @staticmethod
    def pad_image(image_np):
        """Pads the image from (260, 346) to (288, 352).
        Pads Right and Bottom to match standard Deep Learning padding.
        """
        h, w, c = image_np.shape
        target_h, target_w = ImageProcessor.TARGET_SHAPE

        pad_h = target_h - h
        pad_w = target_w - w

        # Pad (Top, Bottom, Left, Right)
        # We pad 0 on Top/Left, and the rest on Bottom/Right
        # This preserves the (0,0) coordinate alignment with events
        padded_image = cv2.copyMakeBorder(
            image_np,
            0,
            pad_h,  # Top, Bottom
            0,
            pad_w,  # Left, Right
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Pad with black
        )

        return padded_image

    def process_all_images(self, input_dir, output_dir):
        """Process all images: Enhance -> Pad -> Save as .npy"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        all_files = list(input_dir.glob("**/*.png"))
        print(f"Found {len(all_files)} images to process.")

        for file_path in tqdm(all_files, desc="Processing Images"):
            # Maintain directory structure
            rel_path = file_path.relative_to(input_dir)
            # Change extension to .npy
            output_file = output_dir / rel_path.with_suffix(".npy")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            self._process_single_file(file_path, output_file)

    def _process_single_file(self, input_path, output_path):
        try:
            # 1. Read Image (OpenCV reads as BGR)
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError("Could not read image")

            # 2. Enhance (CLAHE)
            enhanced = self.enhance_image(image)

            # 3. Pad to (288, 352)
            padded = self.pad_image(enhanced)

            # 4. Convert to Float32 and Normalize to [0, 1]
            # Rearrange to (Channels, Height, Width) for PyTorch
            # Output Shape: (3, 288, 352)
            tensor_np = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

            # 5. Save as .npy
            np.save(output_path, tensor_np)

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
