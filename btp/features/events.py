from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class EventFeatureExtractor(nn.Module):
    """Spatiotemporal Feature Extractor for Event Voxel Grids.

    Strategy:
    1. Collapses Time (T) and Polarity (C) dimensions into Channels.
    2. Applies dynamic padding to handle non-standard sensor resolutions (e.g., 260x346).
    3. Uses a modified ResNet-18 backbone to extract multi-scale features.
    """

    def __init__(self, num_time_bins=4, pretrained=True):
        """Args:
        num_time_bins: Number of temporal bins in the voxel grid (default 4).
        pretrained: Whether to load ImageNet weights for the backbone.

        """
        super().__init__()

        # 1. Calculate Input Channels
        # Input shape: (Batch, Time, Polarity, H, W)
        # We flatten Time and Polarity -> In_Channels = Time * Polarity
        self.in_channels = num_time_bins * 2  # e.g., 4 * 2 = 8 channels

        # 2. Load Backbone (ResNet-18)
        # We use ResNet18 because it is lightweight and preserves spatial info well.
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # 3. Modify First Layer (The "Adapter")
        # Standard ResNet expects 3 channels (RGB). We have 8 channels.
        # We replace the first Conv2d layer.
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )

        # 4. Weight Initialization (Crucial for Transfer Learning)
        # We average the pre-trained RGB weights across our new 8 channels.
        # This gives the model a "head start" on edge detection compared to random init.
        if pretrained:
            with torch.no_grad():
                # Shape: (64, 3, 7, 7) -> Mean over RGB -> (64, 1, 7, 7)
                avg_weight = torch.mean(old_conv.weight, dim=1, keepdim=True)
                # Replicate for 8 channels -> (64, 8, 7, 7)
                self.backbone.conv1.weight.data = avg_weight.repeat(
                    1, self.in_channels, 1, 1
                )

        # 5. Remove Classification Head (We only need features)
        del self.backbone.fc
        del self.backbone.avgpool

    def forward(self, x):
        """Args:
            x: Event Voxel Grid Tensor of shape (Batch, T, C, H, W)
               e.g. (B, 4, 2, 260, 346)

        Returns:
            features: List of 4 feature maps [C2, C3, C4, C5]
                      C2: stride 4
                      C3: stride 8
                      C4: stride 16
                      C5: stride 32

        """
        # 1. Input Check & Reshape
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got {x.shape}")

        B, T, C, H, W = x.shape

        # Collapse T and C into channels -> (B, 8, H, W)
        x = x.view(B, T * C, H, W)

        # 2. Dynamic Padding
        # Deep networks downsample by 32x (stride 32).
        # Input dimensions MUST be divisible by 32 to allow correct upsampling later.
        # 260 -> 288, 346 -> 352
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if pad_h > 0 or pad_w > 0:
            # Pad format: (Left, Right, Top, Bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 3. Forward Pass through ResNet Stages
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # Stride 4

        c2 = self.backbone.layer1(x)  # Stride 4,  64 ch
        c3 = self.backbone.layer2(c2)  # Stride 8,  128 ch
        c4 = self.backbone.layer3(c3)  # Stride 16, 256 ch
        c5 = self.backbone.layer4(c4)  # Stride 32, 512 ch

        # Return pyramid for Multi-Scale Fusion
        return [c2, c3, c4, c5]


# -----------------------------------------------------------------------------
# Utility Function for Verification (Optional usage in main.py)
# -----------------------------------------------------------------------------


def check_feature_extraction(input_dir):
    """Loads a random processed .npy file and runs it through the feature extractor
    to verify shapes and compatibility.
    """
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("**/*.npy"))

    if not npy_files:
        print("No .npy files found to test.")
        return

    # Load one file
    test_file = npy_files[0]
    print(f"\n--- Testing Feature Extraction on {test_file.name} ---")

    try:
        # Load and simulate batch dimension
        raw_tensor = np.load(test_file)  # Shape (T, C, H, W)

        # Convert to Torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = (
            torch.tensor(raw_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        )

        print(f"Input Tensor Shape: {input_tensor.shape}")
        # Expected: (1, 4, 2, 260, 346)

        # Initialize Model
        # Note: We use T=4 based on your preprocessing configuration
        model = EventFeatureExtractor(num_time_bins=input_tensor.shape[1]).to(device)
        model.eval()

        # Inference
        with torch.no_grad():
            features = model(input_tensor)

        print("Feature Pyramid Output:")
        for i, f in enumerate(features):
            print(f"  Scale {i + 1} (Stride {2 ** (i + 2)}): Shape {list(f.shape)}")

        print("✅ Feature Extractor is compatible with Preprocessing pipeline.")
        print(
            "   (Note: Features are usually computed on-the-fly during training, not saved to disk)"
        )

    except Exception as e:
        print(f"❌ Feature Extraction Failed: {e}")
        import traceback

        traceback.print_exc()
