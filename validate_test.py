import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Subset

from btp.data.dataset import BTPDataset
from btp.model import LowLightObjectDetector

# --- Configuration ---
CHECKPOINT = "checkpoints/model_epoch_50.pth"
EVENT_DIR = "data/processed/Processed_Lowlight_event"
IMAGE_DIR = "data/processed/Processed_Lowlight_Images"
LABEL_DIR = "data/raw/Dataset/Annotations"
OUTPUT_DIR = "presentation_images"  # Folder for your slides
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.2  # Your optimal threshold
NMS_THRESHOLD = 0.4
NUM_SAMPLES = 20  # How many images to generate


def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression to remove duplicates."""
    if len(boxes) == 0:
        return []
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    sorted_indices = torch.argsort(scores, descending=True)
    keep_boxes = []

    while sorted_indices.numel() > 0:
        current_idx = sorted_indices[0]
        keep_boxes.append(current_idx.item())
        if sorted_indices.numel() == 1:
            break

        current_box = boxes[current_idx].unsqueeze(0)
        rest_indices = sorted_indices[1:]
        rest_boxes = boxes[rest_indices]

        x1 = torch.max(current_box[:, 0], rest_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], rest_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], rest_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], rest_boxes[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        intersection = inter_w * inter_h

        area_current = (current_box[:, 2] - current_box[:, 0]) * (
            current_box[:, 3] - current_box[:, 1]
        )
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (
            rest_boxes[:, 3] - rest_boxes[:, 1]
        )
        union = area_current + area_rest - intersection

        iou = intersection / (union + 1e-6)
        valid_mask = iou < iou_threshold
        sorted_indices = rest_indices[valid_mask]

    return keep_boxes


def denormalize_image(img_tensor):
    """Convert tensor to numpy image for OpenCV."""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    return (img_np * 255).astype(np.uint8).copy()


def generate_presentation_images():
    print(f"🚀 Generating {NUM_SAMPLES} images for presentation...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load Validation Data (Unseen)
    full_dataset = BTPDataset(EVENT_DIR, IMAGE_DIR, LABEL_DIR)

    if not os.path.exists("validation_indices.json"):
        print(
            "❌ Validation indices not found! Using random samples from full dataset instead."
        )
        val_dataset = full_dataset
    else:
        with open("validation_indices.json") as f:
            val_indices = json.load(f)
        val_dataset = Subset(full_dataset, val_indices)
        print(f"✅ Loaded {len(val_dataset)} unseen validation samples.")

    # 2. Load Model
    model = LowLightObjectDetector(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # 3. Pick Random Indices
    indices = np.random.choice(len(val_dataset), NUM_SAMPLES, replace=False)

    for i, idx in enumerate(indices):
        events, image, targets = val_dataset[idx]

        # Predict
        events_in = events.unsqueeze(0).to(DEVICE)
        image_in = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(events_in, image_in)[0]

        # Decode
        B, C, H, W = preds.shape
        preds_flat = (
            preds.view(1, 3, 8, H, W).permute(0, 1, 3, 4, 2).contiguous().view(-1, 8)
        )

        candidate_boxes = []
        candidate_scores = []

        for j in range(preds_flat.shape[0]):
            score = torch.sigmoid(preds_flat[j, 4]).item()
            if score > CONF_THRESHOLD:
                pred_vals = preds_flat[j]
                p_x = torch.sigmoid(pred_vals[0]).item()
                p_y = torch.sigmoid(pred_vals[1]).item()
                p_w = torch.exp(pred_vals[2]).item()
                p_h = torch.exp(pred_vals[3]).item()

                remainder = j % (H * W)
                grid_y = remainder // W
                grid_x = remainder % W

                cx = (grid_x + p_x) / W
                cy = (grid_y + p_y) / H

                x1 = cx - p_w / 2
                y1 = cy - p_h / 2
                x2 = cx + p_w / 2
                y2 = cy + p_h / 2

                candidate_boxes.append([x1, y1, x2, y2])
                candidate_scores.append(score)

        # NMS
        keep_indices = nms(candidate_boxes, candidate_scores, NMS_THRESHOLD)

        # --- Draw ---
        img_vis = denormalize_image(image)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        height, width, _ = img_vis.shape

        # 1. Ground Truth (GREEN)
        for t in targets:
            gt_x, gt_y, gt_w, gt_h = t[1:]
            x1 = int((gt_x - gt_w / 2) * width)
            y1 = int((gt_y - gt_h / 2) * height)
            x2 = int((gt_x + gt_w / 2) * width)
            y2 = int((gt_y + gt_h / 2) * height)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(img_vis, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Prediction (RED)
        for k in keep_indices:
            box = candidate_boxes[k]
            score = candidate_scores[k]
            px1 = int(box[0] * width)
            py1 = int(box[1] * height)
            px2 = int(box[2] * width)
            py2 = int(box[3] * height)

            cv2.rectangle(img_vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
            # Display score nicely
            label = f"{score:.2f}"
            cv2.putText(
                img_vis,
                label,
                (px1, py1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Save
        save_path = f"{OUTPUT_DIR}/result_{i + 1}.png"
        cv2.imwrite(save_path, img_vis)
        print(f"Saved {save_path}")

    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder for your slides.")


if __name__ == "__main__":
    generate_presentation_images()
