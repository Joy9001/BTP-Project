import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """Simplified YOLO Loss for single-scale or multi-scale.
        We calculate loss primarily on the finest scale (Scale 1) for stability.

        predictions: List of [Scale1, Scale2, Scale3, Scale4] tensors
        targets: (N, 6) -> [batch_idx, cls, x, y, w, h]
        """
        # We focus training on Scale 1 (High resolution)
        # Shape: (B, 24, H, W) where 24 = 3 anchors * (5+3)
        pred = predictions[0]
        B, C, H, W = pred.shape

        # Reshape prediction: (B, 3, H, W, 8) -> (x,y,w,h,conf,c1,c2,c3)
        pred = pred.view(B, 3, 8, H, W).permute(0, 1, 3, 4, 2)

        # Separate components
        # Sigmoid to force 0-1 range
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = torch.exp(pred[..., 2:4])  # Width/Height usually exponential
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])

        # Build Targets
        obj_mask = torch.zeros((B, 3, H, W), device=pred.device, dtype=torch.bool)
        target_box = torch.zeros((B, 3, H, W, 4), device=pred.device)
        target_cls = torch.zeros((B, 3, H, W, 3), device=pred.device)

        for t in targets:
            b_idx, c_idx, gx, gy, gw, gh = t
            b_idx = int(b_idx)

            # Map normalized (0-1) coords to Grid (H, W)
            grid_x = int(gx * W)
            grid_y = int(gy * H)

            if grid_x < W and grid_y < H:
                # Simplified: Assign to all 3 anchors at this cell
                obj_mask[b_idx, :, grid_y, grid_x] = 1

                # Target Box (relative to cell)
                # Ideally tx = gx * W - grid_x, but for simplicity we regress absolute for now
                target_box[b_idx, :, grid_y, grid_x, 0] = gx
                target_box[b_idx, :, grid_y, grid_x, 1] = gy
                target_box[b_idx, :, grid_y, grid_x, 2] = gw
                target_box[b_idx, :, grid_y, grid_x, 3] = gh

                # Target Class (One-hot)
                if int(c_idx) < 3:
                    target_cls[b_idx, :, grid_y, grid_x, int(c_idx)] = 1

        # Calculate Loss
        # 1. Box Loss (only for objects)
        # Note: Simplified regression loss (MSE)
        pred_box_flat = torch.cat([pred_xy, pred_wh], dim=-1)
        loss_box = self.mse(pred_box_flat[obj_mask], target_box[obj_mask])

        # 2. Objectness Loss (All cells)
        # Target confidence is 1 where object exists, 0 otherwise
        target_conf = obj_mask.float()
        loss_conf = self.mse(pred_conf, target_conf)

        # 3. Class Loss
        loss_cls = self.mse(pred_cls[obj_mask], target_cls[obj_mask])

        # Weighted Sum
        return 5.0 * loss_box + 1.0 * loss_conf + 1.0 * loss_cls
