import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ImageVisualizer:
    """A class for visualizing image processing results.

    This class provides methods for:
    - Comparing original vs processed images
    - Plotting intensity histograms
    - Random sampling for visualization
    """

    @staticmethod
    def get_random_images(subdir, num_images=1):
        """Get random images from a subdirectory.

        Args:
            subdir: Path to the subdirectory
            num_images: Number of random images to get

        Returns:
            List of Path objects to random images

        """
        image_files = list(subdir.glob("*.png"))
        if not image_files:
            return []
        return random.sample(image_files, min(num_images, len(image_files)))

    @classmethod
    def visualize_preprocessing(
        cls, input_dir, output_dir, num_examples=5, save_dir=None
    ):
        """Visualize the effects of low-light image preprocessing.

        Args:
            input_dir: Directory containing original low-light images
            output_dir: Directory containing processed images
            num_examples: Number of example images to visualize
            save_dir: Directory to save visualizations (if None, displays instead)

        """
        # Get all subdirectories
        subdirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]

        # Create figure for visualization
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)

        # Process and visualize examples
        for i, subdir in enumerate(subdirs[:num_examples]):
            # Get random image from the subdirectory
            image_files = cls.get_random_images(subdir)
            if not image_files:
                continue

            original_path = image_files[0]
            processed_path = (
                Path(output_dir) / subdir.name / f"processed_{original_path.name}"
            )

            # Load images
            original = Image.open(original_path).convert("RGB")
            processed = Image.open(processed_path).convert("RGB")

            # Plot original and processed images
            axes[i, 0].imshow(original)
            axes[i, 0].set_title(f"Original: {original_path.name}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(processed)
            axes[i, 1].set_title(f"Processed: {processed_path.name}")
            axes[i, 1].axis("off")

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "lowlight_preprocessing_comparison.png")
            print(
                f"Visualization saved to {save_dir / 'lowlight_preprocessing_comparison.png'}"
            )

        plt.show()
        plt.close()

    @classmethod
    def plot_histograms(cls, input_dir, output_dir, num_examples=5, save_dir=None):
        """Plot histograms of original and processed images to show intensity distribution changes."""
        # Get all subdirectories
        subdirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]

        # Create figure for histograms
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)

        # Process and plot histograms
        for i, subdir in enumerate(subdirs[:num_examples]):
            # Get random image from the subdirectory
            image_files = cls.get_random_images(subdir)
            if not image_files:
                continue

            original_path = image_files[0]
            processed_path = (
                Path(output_dir) / subdir.name / f"processed_{original_path.name}"
            )

            # Load images
            original = np.array(Image.open(original_path).convert("RGB"))
            processed = np.array(Image.open(processed_path).convert("RGB"))

            # Plot histograms with adjusted range and bin size
            axes[i, 0].hist(
                original.ravel(), bins=50, range=(0, 100), color="blue", alpha=0.7
            )
            axes[i, 0].set_title(f"Original Histogram: {original_path.name}")
            axes[i, 0].set_xlabel("Pixel Intensity (0-100)")
            axes[i, 0].set_ylabel("Frequency")
            axes[i, 0].set_xlim(0, 100)  # Explicitly set x-axis limits

            axes[i, 1].hist(
                processed.ravel(), bins=50, range=(0, 100), color="red", alpha=0.7
            )
            axes[i, 1].set_title(f"Processed Histogram: {processed_path.name}")
            axes[i, 1].set_xlabel("Pixel Intensity (0-100)")
            axes[i, 1].set_ylabel("Frequency")
            axes[i, 1].set_xlim(0, 100)  # Explicitly set x-axis limits

            # Add grid for better readability
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "lowlight_preprocessing_histograms.png")
            print(
                f"Histograms saved to {save_dir / 'lowlight_preprocessing_histograms.png'}"
            )

        plt.show()
        plt.close()
