from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BTPDataset(Dataset):
    """Loads preprocessed Event (.npy) and Image (.npy) tensors.
    Converts Segmentation Masks (.png) into Bounding Boxes on-the-fly.
    """

    def __init__(self, event_dir, image_dir, label_dir):
        self.event_dir = Path(event_dir)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        # Find all common file IDs
        # We look for .npy files in event_dir and assume matching image/label exist
        self.file_ids = [f.stem for f in self.event_dir.glob("**/*.npy")]
        self.file_ids.sort()

        print(f"📅 Dataset Initialized. Found {len(self.file_ids)} samples.")

    def __len__(self):
        return len(self.file_ids)

    def mask_to_bboxes(self, mask_path):
        """Converts a segmentation mask (PNG) to a list of bounding boxes.
        Format: [class_id, x_center, y_center, width, height] (Normalized 0-1)
        """
        # Load mask (Grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return torch.zeros((0, 5))

        # Image dimensions (Original)
        h_img, w_img = mask.shape

        # Unique object IDs (assuming mask pixel value = class ID or instance ID)
        # In LLE-VOS, usually 0 is background.
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # Exclude background

        boxes = []
        for obj_id in obj_ids:
            # Create binary mask for this object
            pos = np.where(mask == obj_id)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # Convert to XYWH (Normalized)
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2
            y_center = ymin + h / 2

            # Normalize
            x_norm = x_center / w_img
            y_norm = y_center / h_img
            w_norm = w / w_img
            h_norm = h / h_img

            # Class ID (Assuming mapping or just binary object detection)
            # For now, we treat all non-zero pixels as class 0 (Object)
            cls_id = 0

            boxes.append([cls_id, x_norm, y_norm, w_norm, h_norm])

        if len(boxes) == 0:
            return torch.zeros((0, 5))

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        # 1. Load Paths (Using glob to handle potential subdirectories)
        try:
            event_path = next(self.event_dir.glob(f"**/{file_id}.npy"))
            image_path = next(self.image_dir.glob(f"**/{file_id}.npy"))

            # Masks might be .png or .jpg in the downloaded dataset
            label_path = list(self.label_dir.glob(f"**/{file_id}.png"))
            if not label_path:
                label_path = list(self.label_dir.glob(f"**/{file_id}.jpg"))
            label_path = label_path[0] if label_path else None

            # 2. Load Tensors
            evt_tensor = torch.from_numpy(np.load(event_path)).float()
            img_tensor = torch.from_numpy(np.load(image_path)).float()

            # 3. Process Labels
            if label_path and label_path.exists():
                labels = self.mask_to_bboxes(label_path)
            else:
                labels = torch.zeros((0, 5), dtype=torch.float32)

            return evt_tensor, img_tensor, labels

        except Exception as e:
            print(f"Error loading {file_id}: {e}")
            # Return dummy data to prevent crash
            return (
                torch.zeros(4, 2, 260, 346),
                torch.zeros(3, 288, 352),
                torch.zeros((0, 5)),
            )

    @staticmethod
    def collate_fn(batch):
        events, images, labels = zip(*batch, strict=False)
        events = torch.stack(events, 0)
        images = torch.stack(images, 0)

        # Add batch index to labels for Loss calculation
        new_labels = []
        for i, lab in enumerate(labels):
            if lab.shape[0] > 0:
                batch_idx = torch.ones((lab.shape[0], 1)) * i
                lab_with_idx = torch.cat([batch_idx, lab], dim=1)
                new_labels.append(lab_with_idx)

        if len(new_labels) > 0:
            labels = torch.cat(new_labels, 0)
        else:
            labels = torch.zeros((0, 6))

        return events, images, labels
