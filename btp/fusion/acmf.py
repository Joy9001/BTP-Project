import torch
import torch.nn as nn


class ACMFBlock(nn.Module):
    """Adaptive Cross-Modal Fusion (ACMF) Block.

    Fuses features from Image (I) and Events (E) using an Attention Mechanism.
    Formula:
        Attention = Sigmoid(Conv(Concat(I, E)))
        Fused = I * Attention + E * (1 - Attention)

    This allows the network to dynamically select the best modality for each pixel.
    """

    def __init__(self, in_channels):
        super().__init__()

        # 1. Attention Generator
        # Takes concatenation of Image + Event (C + C = 2C channels)
        # Compresses to 1 channel (Spatial Attention Map)
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),  # Reduce dim
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),  # Spatial context
            nn.Sigmoid(),  # Range [0, 1]
        )

        # 2. Feature Refinement
        # Smooths the result after fusion
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, feat_img, feat_event):
        """Args:
        feat_img: (B, C, H, W)
        feat_event: (B, C, H, W)

        """
        # 1. Concatenate along channel dimension
        combined = torch.cat([feat_img, feat_event], dim=1)  # (B, 2C, H, W)

        # 2. Generate Attention Map (B, 1, H, W)
        # Values close to 1.0 -> Trust Image
        # Values close to 0.0 -> Trust Events
        alpha = self.attention_net(combined)

        # 3. Weighted Sum (Fusion)
        fused = (feat_img * alpha) + (feat_event * (1 - alpha))

        # 4. Refine
        out = self.refine_conv(fused)

        return out


class FusionNetwork(nn.Module):
    """Container that holds 4 ACMF Blocks (one for each scale)."""

    def __init__(self):
        super().__init__()

        # ResNet18 Channel counts for the 4 scales:
        # Scale 1: 64 channels
        # Scale 2: 128 channels
        # Scale 3: 256 channels
        # Scale 4: 512 channels

        self.acmf_1 = ACMFBlock(64)
        self.acmf_2 = ACMFBlock(128)
        self.acmf_3 = ACMFBlock(256)
        self.acmf_4 = ACMFBlock(512)

    def forward(self, img_pyramid, evt_pyramid):
        """Args:
        img_pyramid: List of [c2, c3, c4, c5] from Image Branch
        evt_pyramid: List of [c2, c3, c4, c5] from Event Branch

        """
        f1 = self.acmf_1(img_pyramid[0], evt_pyramid[0])
        f2 = self.acmf_2(img_pyramid[1], evt_pyramid[1])
        f3 = self.acmf_3(img_pyramid[2], evt_pyramid[2])
        f4 = self.acmf_4(img_pyramid[3], evt_pyramid[3])

        return [f1, f2, f3, f4]


# -----------------------------------------------------------------------------
# Verification Function
# -----------------------------------------------------------------------------
def check_fusion_module():
    print("\n--- Testing Cross-Modal Fusion ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_net = FusionNetwork().to(device)

    # Create Dummy Pyramids (B=1)
    # Shapes must match what we saw in your verification output
    img_pyr = [
        torch.randn(1, 64, 72, 88).to(device),
        torch.randn(1, 128, 36, 44).to(device),
        torch.randn(1, 256, 18, 22).to(device),
        torch.randn(1, 512, 9, 11).to(device),
    ]

    evt_pyr = [
        torch.randn(1, 64, 72, 88).to(device),
        torch.randn(1, 128, 36, 44).to(device),
        torch.randn(1, 256, 18, 22).to(device),
        torch.randn(1, 512, 9, 11).to(device),
    ]

    try:
        fused_pyr = fusion_net(img_pyr, evt_pyr)
        print("✅ Fusion successful!")
        for i, f in enumerate(fused_pyr):
            print(f"  Fused Scale {i + 1}: {list(f.shape)}")

    except Exception as e:
        print(f"❌ Fusion Failed: {e}")


if __name__ == "__main__":
    check_fusion_module()
