import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm

class ImageFeatureExtractor:
    """Feature extractor for low-light images using Vision Transformer models.

    This class provides functionality for:
    - Loading and preprocessing images from a directory structure.
    - Batch processing with DataLoader for each subdirectory.
    - Feature extraction using pre-trained ViT models.
    - Saving features in a folder-wise manner, with one .npy file per image.
    """

    class _ImageDataset(Dataset):
        """Internal Dataset class for loading images from a list of file paths."""

        def __init__(self, image_paths):
            self.image_paths = image_paths

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                return img, img_path
            except Exception as e:
                print(
                    f"⚠️ Warning: Could not load image {img_path}. Skipping. Error: {e}"
                )
                return None, None

    def __init__(
        self, model_name="google/vit-base-patch16-224-in21k", batch_size=16, device=None
    ):
        """Initialize the image feature extractor."""
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.processor = None
        self.model = None

        print("✓ ImageFeatureExtractor initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")

    def load_model(self):
        """Load the pre-trained model and processor if they are not already loaded."""
        if self.model is None or self.processor is None:
            print("Loading pre-trained model...")
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = (
                AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            )
            print("✓ Model loaded successfully!")

    def process_directory(self, input_dir, output_dir, file_extension="npy"):
        """Extracts features from all images in a directory and saves one feature
        file per image, mirroring the input folder structure.

        Args:
            input_dir (str or Path): The root directory containing image subdirectories.
            output_dir (str or Path): The root directory where feature files will be saved.
            file_extension (str): The file extension for the saved feature files (e.g., 'npy').

        """
        self.load_model()

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subdirectories = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        print(f"Found {len(subdirectories)} subdirectories to process in '{input_dir}'")

        for subdir in tqdm(subdirectories, desc="Processing subdirectories", ncols=100):
            # Create a corresponding output subdirectory
            output_subdir_path = output_dir / subdir.relative_to(input_dir)
            output_subdir_path.mkdir(parents=True, exist_ok=True)

            # Find all image files in the current subdirectory
            image_files = sorted(
                list(subdir.glob("*.png"))
                + list(subdir.glob("*.jpg"))
                + list(subdir.glob("*.jpeg"))
            )

            if not image_files:
                continue

            # Set up a dataset and dataloader for the current subdirectory's images
            dataset = self._ImageDataset(image_files)

            def collate_fn(batch):
                batch = [b for b in batch if b[0] is not None]
                if not batch:
                    return None, None
                images, paths = zip(*batch, strict=False)
                encoding = self.processor(images=list(images), return_tensors="pt")
                return encoding, paths

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=os.cpu_count(),
                pin_memory=True,
            )

            # Process images in batches for efficiency
            file_progress = tqdm(
                dataloader, desc=f"Processing {subdir.name}", leave=False, ncols=100
            )
            for batch_encoding, paths in file_progress:
                if batch_encoding is None:
                    continue

                pixel_values = batch_encoding["pixel_values"].to(self.device)
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)

                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # After processing a batch, iterate through its results to save each one
                for i in range(len(paths)):
                    feature_vector = cls_embeddings[i]
                    original_path = Path(paths[i])

                    # Create an output path with the same name as the image, but a new extension
                    output_file_path = (
                        output_subdir_path / original_path.name
                    ).with_suffix(f".{file_extension}")

                    # Save the single feature vector
                    np.save(output_file_path, feature_vector.astype(np.float16))

        print(
            f"\n🎉 Image feature extraction complete. All features saved in {output_dir}"
        )
