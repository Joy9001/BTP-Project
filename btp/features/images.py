from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageFeatureExtractor(nn.Module):
    """Standard RGB Feature Extractor using ResNet-18.

    Input: (Batch, 3, Height, Width)
           Height, Width should be multiples of 32 (e.g., 288x352).

    Output: List of 4 feature maps [C2, C3, C4, C5] matching the Event Branch.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Load Pretrained Backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # 2. Remove Classification Head
        del self.backbone.fc
        del self.backbone.avgpool

        # Note: We DO NOT modify the first layer because standard images are already 3 channels (RGB).

    def forward(self, x):
        """Args:
            x: Image Tensor (Batch, 3, H, W)

        Returns:
            features: List of 4 feature maps [C2, C3, C4, C5]

        """
        # Stage 1 (Initial Conv + MaxPool) -> Stride 4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Stage 2 (Layer 1) -> Stride 4, 64 channels
        c2 = self.backbone.layer1(x)

        # Stage 3 (Layer 2) -> Stride 8, 128 channels
        c3 = self.backbone.layer2(c2)

        # Stage 4 (Layer 3) -> Stride 16, 256 channels
        c4 = self.backbone.layer3(c3)

        # Stage 5 (Layer 4) -> Stride 32, 512 channels
        c5 = self.backbone.layer4(c4)

        return [c2, c3, c4, c5]


# -----------------------------------------------------------------------------
# Verification Function
# -----------------------------------------------------------------------------
def check_image_features(input_dir):
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("**/*.npy"))

    if not npy_files:
        print("No .npy files found.")
        return

    test_file = npy_files[0]
    print(f"\n--- Testing Image Feature Extraction on {test_file.name} ---")

    try:
        # Load and add Batch Dimension
        raw_tensor = np.load(test_file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = (
            torch.tensor(raw_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        )

        print(f"Input Shape: {input_tensor.shape}")

        # Run Model
        model = ImageFeatureExtractor(pretrained=True).to(device)
        model.eval()

        with torch.no_grad():
            features = model(input_tensor)

        print("Feature Pyramid Output:")
        for i, f in enumerate(features):
            print(f"  Scale {i + 1}: {list(f.shape)}")

        print("✅ Image Branch is ready for Fusion.")

    except Exception as e:
        print(f"❌ Failed: {e}")
