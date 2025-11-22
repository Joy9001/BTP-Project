from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class ImageProcessor:
    """A class for processing low-light images with GPU acceleration.

    This class provides methods for:
    - Image normalization
    - Denoising
    - Batch processing of image directories
    """

    def __init__(self, device=None):
        """Initialize the image processor."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"ImageProcessor initialized on {self.device}")

    @staticmethod
    def normalize_image(image_tensor):
        """Normalize image to [0, 1] range."""
        return (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

    @staticmethod
    def denoise_image(image_tensor, kernel_size=3):
        """Apply simple denoising using median filtering."""
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        denoised = F.avg_pool2d(
            image_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        return denoised.squeeze(0)

    def process_image(self, image_path):
        """Process a single image with normalization and denoising."""
        transform = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(self.device)

        normalized = self.normalize_image(image_tensor)
        denoised = self.denoise_image(normalized)

        if self.device.type == "cuda":
            denoised = denoised.cpu()

        return transforms.ToPILImage()(denoised)

    def process_all_images(self, input_dir, output_dir):
        """Process all images in a directory structure."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

        for subdir in tqdm(subdirs, desc="Processing directories"):
            output_subdir = output_dir / subdir.name
            output_subdir.mkdir(parents=True, exist_ok=True)

            image_files = list(subdir.glob("*.png"))

            for img_path in image_files:
                try:
                    processed_image = self.process_image(img_path)
                    output_path = output_subdir / f"processed_{img_path.name}"
                    processed_image.save(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
