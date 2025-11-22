import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from btp.data.dataset import BTPDataset
from btp.model import LowLightObjectDetector
from btp.training.loss import DetectionLoss

# --- Configuration ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (UPDATE THESE TO YOUR ACTUAL PATHS)
EVENT_DIR = "data/processed/Processed_Lowlight_event"
IMAGE_DIR = "data/processed/Processed_Lowlight_Images"
LABEL_DIR = "data/raw/Dataset/Annotations"  # Path to your .png masks


def train():
    print(f"🚀 Starting Training on {DEVICE}...")

    # 1. Data
    dataset = BTPDataset(EVENT_DIR, IMAGE_DIR, LABEL_DIR)
    if len(dataset) == 0:
        print("❌ No data found! Check your paths.")
        return

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=BTPDataset.collate_fn
    )

    # 2. Model
    model = LowLightObjectDetector(num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = DetectionLoss()

    # 3. Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_idx, (events, images, targets) in enumerate(pbar):
            print(f"Batch {batch_idx + 1}:")
            events = events.to(DEVICE)
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            predictions = model(events, images)

            # Loss
            loss = criterion(predictions, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"📉 Epoch {epoch + 1} Summary: Avg Loss = {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pth")
            print(f"💾 Saved checkpoint to checkpoints/model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()
