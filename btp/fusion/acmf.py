import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention (CA) module.

    This module computes channel-wise attention weights by aggregating spatial
    information using both average and max pooling.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Global Max Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # --- FIX ---
        # Ensure the intermediate channel dimension is at least 1 to avoid
        # creating a layer with 0 output channels when in_planes < ratio.
        hidden_planes = max(1, in_planes // ratio)

        # A small Multi-Layer Perceptron (MLP) to compute attention weights
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply pooling operations
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Sum the outputs and apply sigmoid to get attention weights
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention (SA) module.

    This module generates a spatial attention map by pooling across channels
    and then applying a convolution.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # A convolution layer to process the pooled feature maps
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate the pooled features
        x = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid to get the spatial attention map
        x = self.conv1(x)
        return self.sigmoid(x)


class AdaptiveCrossModalFusion(nn.Module):
    """Adaptive Cross-Modal Fusion (ACMF) module.

    This module adaptively fuses features from image and event modalities
    using channel and spatial attention mechanisms to enhance relevant details
    and suppress noise, as described for low-light VOS.
    """

    def __init__(self, in_channels, out_channels, ratio=16, kernel_size=7):
        super(AdaptiveCrossModalFusion, self).__init__()
        # Attention mechanisms for the event features
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

        # Convolutional layers to refine features after fusion
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, F_img, F_evt):
        """Forward pass for the ACMF module.

        Args:
            F_img (torch.Tensor): Image features of shape (B, C, H, W).
            F_evt (torch.Tensor): Event features of shape (B, C, H, W).

        Returns:
            torch.Tensor: Fused and refined features.

        """
        # --- Step 1: Extract edge/structural info from events using attention ---
        # Apply channel attention to event features
        F_evt_ca = self.ca(F_evt) * F_evt
        # Apply spatial attention to event features
        F_evt_sa = self.sa(F_evt) * F_evt

        # --- Step 2: Cross-modal interaction and refinement ---
        # Element-wise multiplication to fuse image features with attended event features
        F_fused_ca = self.conv1(F_img * F_evt_ca)
        F_fused_sa = self.conv2(F_img * F_evt_sa)

        F_fused_ca = F.relu(self.bn1(F_fused_ca))
        F_fused_sa = F.relu(self.bn2(F_fused_sa))

        # --- Step 3: Final output generation ---
        # Sum the two enhanced features for a robust final representation
        F_out = F_fused_ca + F_fused_sa

        return F_out
