import torch.nn as nn


class YOLOHead(nn.Module):
    """Multi-Scale Detection Head.

    For each feature map pixel, it predicts:
    - Bounding Box offsets (dx, dy, dw, dh)
    - Objectness Score (Is there an object?)
    - Class Scores (Car, Pedestrian, Bike)
    """

    def __init__(self, num_classes=3, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Output channels per anchor = (4 box coords + 1 obj score + num_classes)
        self.output_dim = num_anchors * (5 + num_classes)

        # We need a separate prediction layer for each scale channel count
        # Scale 1 (64 ch), Scale 2 (128 ch), Scale 3 (256 ch), Scale 4 (512 ch)
        self.head_1 = self._make_head(64, self.output_dim)
        self.head_2 = self._make_head(128, self.output_dim)
        self.head_3 = self._make_head(256, self.output_dim)
        self.head_4 = self._make_head(512, self.output_dim)

    def _make_head(self, in_channels, out_channels):
        """Standard Conv Block for Detection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels * 2, out_channels, kernel_size=1
            ),  # 1x1 Conv for prediction
        )

    def forward(self, fused_features):
        """Args:
            fused_features: List of 4 feature maps [f1, f2, f3, f4]

        Returns:
            predictions: List of 4 raw prediction tensors

        """
        p1 = self.head_1(fused_features[0])
        p2 = self.head_2(fused_features[1])
        p3 = self.head_3(fused_features[2])
        p4 = self.head_4(fused_features[3])

        return [p1, p2, p3, p4]
