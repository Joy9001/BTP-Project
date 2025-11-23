import json
import os

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from btp.data.dataset import BTPDataset
from btp.model import LowLightObjectDetector

# --- Configuration ---
CHECKPOINT = "checkpoints/model_epoch_50.pth"
EVENT_DIR = "data/processed/Processed_Lowlight_event"
IMAGE_DIR = "data/processed/Processed_Lowlight_Images"
LABEL_DIR = "data/raw/Dataset/Annotations"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.2  # Keep low to let NMS do the work
NMS_IOU_THRESHOLD = 0.4  # Merging threshold
EVAL_IOU_THRESHOLD = 0.4  # Correctness threshold


def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def evaluate():
    print(f"🚀 Starting Corrected Evaluation on {DEVICE}...")
    print(
        f"Using confidence threshold: {CONF_THRESHOLD}, NMS IoU threshold: {NMS_IOU_THRESHOLD}, Eval IoU threshold: {EVAL_IOU_THRESHOLD}"
    )

    # 1. Load Data
    full_dataset = BTPDataset(EVENT_DIR, IMAGE_DIR, LABEL_DIR)

    if not os.path.exists("validation_indices.json"):
        print("❌ No validation split found! Run train.py first.")
        return

    with open("validation_indices.json") as f:
        val_indices = json.load(f)

    val_dataset = Subset(full_dataset, val_indices)
    print(f"🔍 Evaluating on {len(val_dataset)} samples with NMS.")

    loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=BTPDataset.collate_fn
    )

    # 2. Load Model
    model = LowLightObjectDetector(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # Statistics
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for events, images, targets in tqdm(loader):
            events = events.to(DEVICE)
            images = images.to(DEVICE)

            # Predict
            preds = model(events, images)[0]
            B, C, H, W = preds.shape
            preds_flat = (
                preds.view(B, 3, 8, H, W)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
                .view(-1, 8)
            )

            # --- 1. Collect Candidates ---
            conf_scores = torch.sigmoid(preds_flat[:, 4])
            # Filter out absolute garbage (<0.001) to speed up NMS
            mask = conf_scores > CONF_THRESHOLD

            pred_boxes_tensor = []
            pred_scores_tensor = []

            if mask.any():
                valid_preds = preds_flat[mask]
                valid_scores = conf_scores[mask]

                for i in range(valid_preds.shape[0]):
                    p = valid_preds[i]
                    # Decode
                    p_x = torch.sigmoid(p[0]).item()
                    p_y = torch.sigmoid(p[1]).item()
                    p_w = torch.exp(p[2]).item()
                    p_h = torch.exp(p[3]).item()

                    # Grid recovery logic
                    original_idx = torch.where(mask)[0][i]
                    remainder = original_idx % (H * W)
                    grid_y = remainder // W
                    grid_x = remainder % W

                    cx = (grid_x + p_x) / W
                    cy = (grid_y + p_y) / H

                    x1 = max(0.0, cx - p_w / 2)
                    y1 = max(0.0, cy - p_h / 2)
                    x2 = min(1.0, cx + p_w / 2)
                    y2 = min(1.0, cy + p_h / 2)

                    pred_boxes_tensor.append([x1, y1, x2, y2])
                    pred_scores_tensor.append(valid_scores[i].item())

            # --- 2. Apply NMS ---
            final_pred_boxes = []
            if len(pred_boxes_tensor) > 0:
                boxes_t = torch.tensor(pred_boxes_tensor, device=DEVICE)
                scores_t = torch.tensor(pred_scores_tensor, device=DEVICE)

                # PyTorch's built-in NMS is very fast
                keep_indices = torchvision.ops.nms(boxes_t, scores_t, NMS_IOU_THRESHOLD)

                final_pred_boxes = boxes_t[keep_indices].tolist()

            # --- 3. Evaluate against Ground Truth ---
            gt_boxes = []
            for t in targets:
                gt_cx, gt_cy, gt_w, gt_h = t[2:]
                x1 = gt_cx - gt_w / 2
                y1 = gt_cy - gt_h / 2
                x2 = gt_cx + gt_w / 2
                y2 = gt_cy + gt_h / 2
                gt_boxes.append([x1, y1, x2, y2])

            matched_gt = set()

            for p_box in final_pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for i, g_box in enumerate(gt_boxes):
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou >= EVAL_IOU_THRESHOLD and best_gt_idx not in matched_gt:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_fn += len(gt_boxes) - len(matched_gt)

    # Metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n📊 --- Final Corrected Results ---")
    print(f"TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
    print("-" * 30)
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔍 Recall:    {recall:.4f}")
    print(f"⭐ F1-Score:  {f1_score:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    evaluate()
