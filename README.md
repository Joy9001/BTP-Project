# Robust Object Detection in Low-Light Scenarios using Event-Visual Fusion

## 📌 Project Overview

This project implements a **Dual-Stream Deep Learning Architecture** to detect objects in extreme low-light conditions. Standard cameras fail in the dark due to noise and motion blur. We solve this by fusing:

1. **Low-Light RGB Frames:** Enhanced via CLAHE for texture context.
2. **Event Camera Data:** High-temporal resolution streams $(x, y, t, p)$ converted to Voxel Grids for motion context.

The model uses a **ResNet-18** backbone for feature extraction and a custom **Adaptive Cross-Modal Fusion (ACMF)** module to dynamically weigh the best sensor modality.

## 🛠️ Setup & Installation

### 1. Prerequisites

* Python >= 3.13
* CUDA-enabled GPU (Recommended for training)

### 2. Installation

We use `uv` for dependency management, but standard `pip` works too.

**Option A: Using `uv` (Recommended)**

```bash
# Sync dependencies from pyproject.toml
uv sync
````

**Option B: Using `pip`**

```bash
pip install -r requirements.txt
# Or manually:
pip install torch torchvision numpy opencv-python tqdm matplotlib kagglehub h5py transformers
```

## 📂 Dataset Preparation

We use the **LLE-VOS** dataset (Low-Light Event Video Object Segmentation). The pipeline expects the following directory structure in `data/raw/`:

```bash
data/raw/Dataset/
├── Lowlight_event/       # Raw .npy event files
├── Lowlight_Images/      # Raw .png low-light images
└── Annotations/          # Ground Truth Masks (.png)
```

## 🚀 Usage Pipeline

### 1\. Preprocessing

Convert raw event streams into Voxel Grids and enhance low-light images.

```bash
# Process both Events and Images
uv run main.py --step all

# Or run individually:
uv run main.py --step preprocess_events
uv run main.py --step preprocess_images
```

* **Output:** Processed tensors saved to `data/processed/`.

### 2\. Training

Train the Dual-Stream Object Detector.

* **Config:** Edit `train.py` to change `BATCH_SIZE` (default: 4) or `NUM_EPOCHS` (default: 50).
* **Split:** Automatically splits data into **80% Train** and **20% Validation**.

<!-- end list -->

```bash
uv run train.py
```

* **Checkpoints:** Saved to `checkpoints/model_epoch_X.pth`.
* **Logs:** Training loss saved to `logs/run3.log`.

### 3\. Evaluation (Metrics)

Calculate **Precision**, **Recall**, and **F1-Score** on the unseen Validation Set.
This script uses **Non-Maximum Suppression (NMS)** to ensure accurate scoring.

```bash
uv run evaluate.py
```

* **Expected Result:** High Recall (\~0.95) indicates the model successfully finds objects in the dark.

### 4\. Visualization (Demo)

Generate visual results for presentations. This creates images with **Green (Ground Truth)** and **Red (Prediction)** bounding boxes overlaid.

```bash
uv run validate_test.py
```

* **Output:** Images saved to `presentation_images/`.

## 🏗️ Project Structure

```bash
joy9001-btp-project/
├── btp/
│   ├── data/           # Dataset loading & Mask-to-BBox conversion
│   ├── detection/      # YOLO-style Detection Head
│   ├── features/       # ResNet-18 Feature Extractors (Event & Image)
│   ├── fusion/         # ACMF Fusion Module
│   ├── processing/     # Voxelization & CLAHE logic
│   └── training/       # Custom Loss Function
├── checkpoints/        # Saved models
├── data/               # Raw & Processed data
├── evaluate.py         # Metrics calculation script
├── main.py             # CLI entry point
├── train.py            # Training loop
└── validate_test.py    # Visualization script
```

## 📊 Results Summary

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Recall** | **94.95%** | Excellent detection rate; effectively "sees" in the dark. |
| **Precision** | **47.96%** | Moderate false alarm rate due to event sensor noise. |
| **F1-Score** | **0.6373** | Strong overall performance for a custom implementation. |

-----

**Note:** If you encounter "No validation split found" errors, ensure you run `train.py` at least once to generate the `validation_indices.json` file.
