"""Deprecated Prediction Script."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from btp.data.dataset import BTPDataset
from btp.model import LowLightObjectDetector

# --- Config ---
# Tries to find the latest checkpoint automatically if specific one isn't found
CHECKPOINT = "checkpoints/model_epoch_50.pth"
EVENT_DIR = "data/processed/Processed_Lowlight_event"
IMAGE_DIR = "data/processed/Processed_Lowlight_Images"
LABEL_DIR = "data/raw/Dataset/Annotations"
OUTPUT_DIR = "visualizations/predictions/run2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression (NMS) to remove overlapping boxes.
    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,) confidence scores
    """
    if len(boxes) == 0:
        return []

    # Convert to tensor if needed
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    # Sort by confidence
    sorted_indices = torch.argsort(scores, descending=True)
    keep_boxes = []

    while sorted_indices.numel() > 0:
        current_idx = sorted_indices[0]
        keep_boxes.append(current_idx.item())

        if sorted_indices.numel() == 1:
            break

        # Compute IoU with the rest
        current_box = boxes[current_idx].unsqueeze(0)
        rest_indices = sorted_indices[1:]
        rest_boxes = boxes[rest_indices]

        # --- IoU Calculation ---
        # Intersection coordinates
        x1 = torch.max(current_box[:, 0], rest_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], rest_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], rest_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], rest_boxes[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        intersection = inter_w * inter_h

        # Union
        area_current = (current_box[:, 2] - current_box[:, 0]) * (
            current_box[:, 3] - current_box[:, 1]
        )
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (
            rest_boxes[:, 3] - rest_boxes[:, 1]
        )
        union = area_current + area_rest - intersection

        iou = intersection / (union + 1e-6)

        # Keep boxes with IoU < threshold (distinct objects)
        valid_mask = iou < iou_threshold
        sorted_indices = rest_indices[valid_mask]

    return keep_boxes


def denormalize_image(img_tensor):
    """Convert normalized float tensor (C,H,W) to uint8 numpy (H,W,C)"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    return (img_np * 255).astype(np.uint8).copy()


def visualize_prediction():
    # 1. Setup
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    dataset = BTPDataset(EVENT_DIR, IMAGE_DIR, LABEL_DIR)

    # Pick a random sample
    idx = np.random.randint(0, len(dataset))
    events, image, targets = dataset[idx]
    file_id = dataset.file_ids[idx]

    print(f"--- Visualizing Sample {idx} (ID: {file_id}) ---")

    # 2. Load Model
    model = LowLightObjectDetector(num_classes=3).to(DEVICE)

    # Check for checkpoint, fallback to latest if specified not found
    checkpoint_path = CHECKPOINT
    if not os.path.exists(checkpoint_path) and os.path.exists("checkpoints"):
        chkpts = sorted(list(Path("checkpoints").glob("model_epoch_*.pth")))
        if chkpts:
            checkpoint_path = str(chkpts[-1])

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print(
            "⚠️ Checkpoint not found! Using random weights (Predictions will be garbage)."
        )

    model.eval()

    # 3. Predict
    events_in = events.unsqueeze(0).to(DEVICE)
    image_in = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(events_in, image_in)[0]  # Scale 1

    # 4. Prepare Images
    base_img = denormalize_image(image)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
    H, W, _ = base_img.shape

    img_raw = base_img.copy()
    img_gt = base_img.copy()
    img_pred = base_img.copy()

    # 5. Draw Ground Truth (Green)
    print(f"Ground Truth Objects: {len(targets)}")
    for t in targets:
        # t is [cls, x, y, w, h] (Normalized)
        gt_x, gt_y, gt_w, gt_h = t[1:]

        x1 = int((gt_x - gt_w / 2) * W)
        y1 = int((gt_y - gt_h / 2) * H)
        x2 = int((gt_x + gt_w / 2) * W)
        y2 = int((gt_y + gt_h / 2) * H)

        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_gt, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # 6. Decode, Filter & Draw Prediction (Red)
    # Reshape to list of all grid cells: (B, 3*H*W, 8)
    B_batch, C_channels, Grid_H, Grid_W = preds.shape
    preds_flat = (
        preds.view(1, 3, 8, Grid_H, Grid_W)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(-1, 8)
    )

    candidate_boxes = []  # [x1, y1, x2, y2]
    candidate_scores = []

    print("Processing predictions for NMS...")

    # Iterate through all grid cells to find candidates > threshold
    for i in range(preds_flat.shape[0]):
        score = torch.sigmoid(preds_flat[i, 4]).item()

        if score > 0.5:  # Confidence Threshold
            pred_vals = preds_flat[i]

            # Decode relative to grid cell
            p_x = torch.sigmoid(pred_vals[0]).item()
            p_y = torch.sigmoid(pred_vals[1]).item()
            p_w = torch.exp(pred_vals[2]).item()
            p_h = torch.exp(pred_vals[3]).item()

            # Recover Grid Position
            remainder = i % (Grid_H * Grid_W)
            grid_y = remainder // Grid_W
            grid_x = remainder % Grid_W

            # Project back to Normalized Image Coordinates [0, 1]
            # abs_cx = (grid_x + offset) / Grid_Width
            abs_cx = (grid_x + p_x) / Grid_W
            abs_cy = (grid_y + p_y) / Grid_H

            # Convert Center-WH to Corners [x1, y1, x2, y2] (Normalized)
            # Note: p_w and p_h are relative to anchor/image depending on loss logic.
            # Assuming logic aligns with dataset.py normalization.
            x1 = abs_cx - p_w / 2
            y1 = abs_cy - p_h / 2
            x2 = abs_cx + p_w / 2
            y2 = abs_cy + p_h / 2

            candidate_boxes.append([x1, y1, x2, y2])
            candidate_scores.append(score)

    # 7. Apply NMS
    if len(candidate_boxes) > 0:
        # Run NMS (IoU threshold 0.4 means overlap > 40% is merged)
        keep_indices = nms(candidate_boxes, candidate_scores, iou_threshold=0.4)
        print(
            f"NMS: Kept {len(keep_indices)} out of {len(candidate_boxes)} candidates."
        )

        for idx in keep_indices:
            box = candidate_boxes[idx]
            score = candidate_scores[idx]

            # Scale to image size for drawing
            px1 = int(box[0] * W)
            py1 = int(box[1] * H)
            px2 = int(box[2] * W)
            py2 = int(box[3] * H)

            print(f"Pred: Conf={score:.4f} Box=[{px1}, {py1}, {px2}, {py2}]")

            cv2.rectangle(img_pred, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(
                img_pred,
                f"{score:.2f}",
                (px1, py1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
    else:
        print("No predictions above 0.5 confidence threshold.")

    # 8. Save files
    sample_dir = Path(OUTPUT_DIR) / file_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    raw_path = sample_dir / f"sample_{file_id}_raw.png"
    gt_path = sample_dir / f"sample_{file_id}_gt.png"
    pred_path = sample_dir / f"sample_{file_id}_pred.png"
    combined_path = sample_dir / f"sample_{file_id}_combined.png"

    # cv2.imwrite(str(raw_path), img_raw)
    # cv2.imwrite(str(gt_path), img_gt)
    # cv2.imwrite(str(pred_path), img_pred)

    # Save a combined view for easy comparison
    combined = np.hstack((img_gt, img_pred))
    cv2.imwrite(str(combined_path), combined)

    print("\n✅ Images Saved:")
    print(f"Output Folder: {sample_dir}")


if __name__ == "__main__":
    visualize_prediction()
