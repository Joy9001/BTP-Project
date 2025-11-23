import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from btp.data.dataset import BTPDataset
from btp.model import LowLightObjectDetector
from btp.training.loss import DetectionLoss

# --- Configuration ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (UPDATE THESE TO YOUR ACTUAL PATHS)
EVENT_DIR = Path.cwd() / "data" / "processed" / "Processed_Lowlight_event"
IMAGE_DIR = Path.cwd() / "data" / "processed" / "Processed_Lowlight_Images"

# Path to your .png masks
LABEL_DIR = Path.cwd() / "data" / "raw" / "Dataset" / "Annotations"

RUN = 3
LOG_PATH = Path.cwd() / "logs" / f"run{RUN}.log"


def train():
    print(f"🚀 Starting Training on {DEVICE}...")

    # 1. Load Full Data
    full_dataset = BTPDataset(EVENT_DIR, IMAGE_DIR, LABEL_DIR)
    total_size = len(full_dataset)

    # 2. Create Split (80% Train, 20% Val)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Fix generator seed for reproducibility (Important!)
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"📊 Data Split: {train_size} Training | {val_size} Validation")

    # SAVE VALIDATION INDICES for evaluate.py
    # We need to know exactly which files are in the validation set later
    val_indices = val_set.indices
    with open("validation_indices.json", "w") as f:
        json.dump(val_indices, f)
    print("💾 Saved validation indices to 'validation_indices.json'")

    # 3. Loaders
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=BTPDataset.collate_fn
    )

    # (Optional) Val loader to monitor progress during training
    # val_loader = DataLoader(
    #     val_set, batch_size=1, shuffle=False, collate_fn=BTPDataset.collate_fn
    # )

    # 4. Model & Loss
    model = LowLightObjectDetector(num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = DetectionLoss()

    # 5. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")
        for batch_idx, (events, images, targets) in enumerate(pbar):
            print(f"Batch {batch_idx + 1}:")
            events = events.to(DEVICE)
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(events, images)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 Epoch {epoch + 1} Summary: Avg Training Loss = {avg_loss:.4f}")
        with open(LOG_PATH, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}\n")

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pth")
            print(f"💾 Saved checkpoint to checkpoints/model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()
