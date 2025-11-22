from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
from .acmf import AdaptiveCrossModalFusion

def fuse_features_in_directories(
    image_features_dir,
    event_features_dir,
    output_dir,
    image_feature_dim=768,
    event_feature_dim=256,
):
    """Loads, fuses, and saves image and event features file-by-file.

    Args:
        image_features_dir (str): Path to the root directory of extracted image features (.npy).
        event_features_dir (str): Path to the root directory of extracted event features (.h5).
        output_dir (str): Path to the directory where fused features will be saved.
        image_feature_dim (int): The dimension of the image feature vectors.
        event_feature_dim (int): The dimension of the event feature vectors.

    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting feature fusion process on device: {device}")

    # The reshaped feature map will have 1 channel.
    in_channels = 1
    out_channels = 1

    # Instantiate the ACMF module and move it to the selected device
    acmf_module = AdaptiveCrossModalFusion(
        in_channels=in_channels, out_channels=out_channels
    ).to(device)
    acmf_module.eval()  # Set to evaluation mode

    # Define a linear projection layer to map event features to image feature dimension
    event_projection = nn.Linear(event_feature_dim, image_feature_dim).to(device)
    event_projection.eval()

    # Define spatial dimensions for reshaping the 768-dim vector. 24 * 32 = 768
    H, W = 24, 32
    if H * W != image_feature_dim:
        raise ValueError(
            f"Incompatible dimensions: {H}x{W} does not match feature dim {image_feature_dim}"
        )

    image_dir = Path(image_features_dir)
    event_dir = Path(event_features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a sorted list of subdirectories (e.g., '00001', '00002', ...)
    subdirectories = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subdirectories)} subdirectories to process.")

    # --- Processing Loop ---
    for subdir in tqdm(subdirectories, desc="Fusing Subdirectories"):
        image_subdir = image_dir / subdir.name
        event_subdir = event_dir / subdir.name
        output_subdir = output_dir / subdir.name
        output_subdir.mkdir(exist_ok=True)

        if not event_subdir.exists():
            print(
                f"⚠️ Warning: Event subdir {event_subdir} not found. Skipping {subdir.name}."
            )
            continue

        image_files = sorted(image_subdir.glob("*.npy"))
        event_files = sorted(event_subdir.glob("*.h5"))

        if len(image_files) != len(event_files):
            print(
                f"⚠️ Warning: Mismatch in file count for {subdir.name}. "
                f"Images: {len(image_files)}, Events: {len(event_files)}. Taking the minimum."
            )
            if len(image_files) > len(event_files):
                image_files = image_files[: len(event_files)]
            else:
                event_files = event_files[: len(image_files)]

        file_progress = tqdm(
            zip(image_files, event_files, strict=False),
            total=len(image_files),
            desc=f"Fusing {subdir.name}",
            leave=False,
        )
        for img_file, evt_file in file_progress:
            try:
                # --- Load and Preprocess ---
                img_feat = np.load(img_file)
                with h5py.File(evt_file, "r") as hf:
                    evt_feat_raw = hf["features"][:]

                if evt_feat_raw.ndim > 1:
                    evt_feat = np.mean(evt_feat_raw, axis=0)
                else:
                    evt_feat = evt_feat_raw

                # --- Dimension Check and Projection ---
                if img_feat.shape[0] != image_feature_dim:
                    print(
                        f"Skipping {img_file.name}, unexpected image dim: {img_feat.shape[0]}"
                    )
                    continue
                if evt_feat.shape[0] != event_feature_dim:
                    print(
                        f"Skipping {evt_file.name}, unexpected event dim: {evt_feat.shape[0]}"
                    )
                    continue

                # Convert event feature to tensor and project it
                evt_tensor_1d = torch.from_numpy(evt_feat).float().to(device)
                with torch.no_grad():
                    evt_feat_projected = event_projection(evt_tensor_1d)

                # Reshape both to (1, 1, H, W)
                img_tensor = (
                    torch.from_numpy(img_feat)
                    .float()
                    .reshape(1, in_channels, H, W)
                    .to(device)
                )
                evt_tensor = evt_feat_projected.reshape(1, in_channels, H, W)

                # --- Fuse Features ---
                with torch.no_grad():
                    fused_output = acmf_module(img_tensor, evt_tensor)

                # --- Save Fused Feature ---
                fused_np = fused_output.cpu().numpy().flatten()

                output_path = output_subdir / img_file.name
                np.save(output_path, fused_np)

            except Exception as e:
                print(
                    f"❌ Error processing file pair: {img_file.name}, {evt_file.name}. Error: {e}"
                )

    print("\n🎉 Feature fusion completed successfully!")
