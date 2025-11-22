import torch
import torch.nn as nn

from btp.detection.head import YOLOHead

# Import the components we built
from btp.features.events import EventFeatureExtractor
from btp.features.images import ImageFeatureExtractor
from btp.fusion.acmf import FusionNetwork


class LowLightObjectDetector(nn.Module):
    """The Complete BTP Model Architecture.

    Flow:
    1. Events -> Event Branch -> Feature Pyramid (E)
    2. Images -> Image Branch -> Feature Pyramid (I)
    3. (E, I) -> Fusion Module -> Fused Pyramid (F)
    4. F -> Detection Head -> Bounding Boxes
    """

    def __init__(self, num_classes=3):
        super().__init__()

        print("🏗️ Building Low-Light Object Detector...")

        # 1. Encoders
        self.event_branch = EventFeatureExtractor(pretrained=True)
        self.image_branch = ImageFeatureExtractor(pretrained=True)

        # 2. Fusion
        self.fusion = FusionNetwork()

        # 3. Detection Head
        # Assuming 3 classes (Car, Pedestrian, Bike)
        self.head = YOLOHead(num_classes=num_classes)

        print("✅ Model Built Successfully.")

    def forward(self, event_voxel, image_tensor):
        """Args:
        event_voxel: (B, 4, 2, 260, 346) - Raw Event Grid
        image_tensor: (B, 3, 288, 352) - Preprocessed Image

        """
        # 1. Extract Features
        # Event Branch handles the dynamic padding internally for 260x346 input
        evt_features = self.event_branch(event_voxel)
        img_features = self.image_branch(image_tensor)

        # 2. Fuse Features
        fused_features = self.fusion(img_features, evt_features)

        # 3. Predict
        predictions = self.head(fused_features)

        return predictions


# -----------------------------------------------------------------------------
# Verification (Run this file directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Instantiate Model
    model = LowLightObjectDetector().to(device)

    # 2. Create Dummy Inputs
    # Note: Event input is unpadded (260x346), Image input is padded (288x352)
    # This simulates exactly what your Dataloader will provide.
    dummy_event = torch.randn(1, 4, 2, 260, 346).to(device)
    dummy_image = torch.randn(1, 3, 288, 352).to(device)

    print("\n--- Starting Full Model Pass ---")
    try:
        outputs = model(dummy_event, dummy_image)

        print("✅ Forward pass successful!")
        print("Output Shapes (Batch, Channels, Grid_H, Grid_W):")
        for i, out in enumerate(outputs):
            print(f"  Scale {i + 1}: {list(out.shape)}")

        print("\nNote: Channels = 3 anchors * (5 + 3 classes) = 24")

    except Exception as e:
        print(f"❌ Model Failed: {e}")
        import traceback

        traceback.print_exc()
