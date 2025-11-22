import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class EventTransformerComponents:
    """Container class for all transformer-based event processing components."""

    class EventTokenizer(nn.Module):
        """Tokenizer for converting event data into embeddings."""

        def __init__(self, d_model=256, max_spatial_dim=(240, 180)):
            super().__init__()
            self.d_model = d_model
            self.max_spatial_dim = max_spatial_dim
            self.spatial_embedding = nn.Embedding(
                max_spatial_dim[0] * max_spatial_dim[1], d_model // 2
            )
            self.temporal_embedding = nn.Linear(1, d_model // 4)
            self.polarity_embedding = nn.Embedding(2, d_model // 4)
            self.token_projection = nn.Linear(d_model, d_model)

        def forward(self, events):
            batch_size, num_events, _ = events.shape
            x, y = events[:, :, 0].long(), events[:, :, 1].long()
            t = events[:, :, 2].unsqueeze(-1)
            p = events[:, :, 3].long()

            spatial_pos = y * self.max_spatial_dim[0] + x
            spatial_embedding = self.spatial_embedding(spatial_pos)
            temporal_embedding = self.temporal_embedding(t)
            polarity_embedding = self.polarity_embedding(p)

            token_embeddings = torch.cat(
                [spatial_embedding, temporal_embedding, polarity_embedding], dim=-1
            )
            token_embeddings = self.token_projection(token_embeddings)
            return token_embeddings

    class EventPositionalEncoding(nn.Module):
        """Positional encoding for event sequences."""

        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, x):
            seq_len = x.size(1)
            device = x.device
            position = torch.arange(seq_len, device=device).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=device).float()
                * (-math.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(1, seq_len, self.d_model, device=device)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            return x + pe

    class EventTransformerEncoder(nn.Module):
        """Complete transformer encoder for event data processing."""

        def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            max_spatial_dim=(240, 180),
        ):
            super().__init__()
            self.tokenizer = EventTransformerComponents.EventTokenizer(
                d_model, max_spatial_dim
            )
            self.pos_encoder = EventTransformerComponents.EventPositionalEncoding(
                d_model
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_encoder_layers
            )
            self.pooling = nn.AdaptiveAvgPool1d(1)

        def forward(self, events, padding_mask=None):
            token_embeddings = self.tokenizer(events)
            token_embeddings = self.pos_encoder(token_embeddings)
            if padding_mask is not None:
                features = self.transformer_encoder(
                    token_embeddings, src_key_padding_mask=padding_mask
                )
            else:
                features = self.transformer_encoder(token_embeddings)
            features = features.transpose(1, 2)  # (B, D, N)
            pooled = self.pooling(features).squeeze(-1)  # (B, D)
            return pooled


class OptimizedEventFeatureExtractor:
    """Optimized GPU-accelerated feature extractor for event data.

    This class uses transformer-based models to extract high-dimensional
    features from event data with memory optimization and batch processing.

    Features:
    - GPU acceleration with CUDA
    - Memory-optimized streaming processing
    - Batch processing with progress tracking
    - Flexible output formats (H5, NPY)
    - Comprehensive directory processing
    """

    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_spatial_dim=(240, 180),
        device=None,
        batch_size=512,
    ):
        """Initialize the feature extractor with specified parameters."""
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required. No GPU available.")
            self.device = torch.device("cuda")
        else:
            if not device.startswith("cuda"):
                raise ValueError("Only CUDA devices are supported.")
            self.device = device

        self.encoder = EventTransformerComponents.EventTransformerEncoder(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dropout=dropout,
            max_spatial_dim=max_spatial_dim,
        ).to(self.device)

        self.batch_size = batch_size
        self.model_dim = model_dim

        print(f"✓ Initialized on {self.device} with batch size {batch_size}")

    def extract_features_stream(self, event_data, max_events=None):
        """Extract features using streaming approach for memory efficiency."""
        if max_events is not None and event_data.shape[0] > max_events:
            indices = np.random.choice(event_data.shape[0], max_events, replace=False)
            event_data = event_data[indices]

        total_events = event_data.shape[0]
        num_batches = math.ceil(total_events / self.batch_size)
        features_np = np.zeros((num_batches, self.model_dim), dtype=np.float16)

        progress_bar = tqdm(
            range(0, total_events, self.batch_size),
            desc="Extracting features",
            ncols=80,
            leave=False,
            mininterval=1.0,
        )

        for i, idx in enumerate(progress_bar):
            end_idx = min(idx + self.batch_size, total_events)
            batch_data = event_data[idx:end_idx]
            batch_tensor = (
                torch.tensor(batch_data, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                pooled_features = self.encoder(batch_tensor)
            features_np[i] = pooled_features.cpu().numpy().astype(np.float16)
        return features_np

    def process_file_to_h5(
        self, input_file, output_file, max_events=None, compression="gzip"
    ):
        """Process a single file and save features in H5 format."""
        try:
            event_data = np.load(input_file)
            features = self.extract_features_stream(event_data, max_events)
            with h5py.File(output_file, "w") as h5f:
                h5f.create_dataset(
                    "features",
                    data=features,
                    dtype="float16",
                    compression=compression,
                    compression_opts=9,
                    chunks=(min(self.batch_size, features.shape[0]), self.model_dim),
                )
            input_size = os.path.getsize(input_file) / 1e6
            output_size = os.path.getsize(output_file) / 1e6
            print(
                f"✓ Saved to {output_file} ({output_size:.2f} MB, compression ratio: {input_size / output_size:.2f}x)"
            )
            del features, event_data
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def process_directory(
        self,
        input_dir,
        output_dir,
        max_events=None,
        compression="gzip",
        file_extension="h5",
        glob="**/*.npy",
    ):
        """Process all files in a directory structure."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_files = list(input_dir.glob(glob))
        print(f"Found {len(all_files)} files to process")

        for file_path in tqdm(
            all_files, desc="Processing files", ncols=80, leave=False
        ):
            rel = file_path.relative_to(input_dir)
            out = output_dir / rel.with_suffix(f".{file_extension}")
            out.parent.mkdir(parents=True, exist_ok=True)
            self.process_file_to_h5(str(file_path), str(out), max_events, compression)
        print(f"✓ Processing complete. Features saved in {output_dir}")

    def process_subdirectory(
        self,
        input_subdir,
        output_subdir,
        max_events=None,
        compression="gzip",
        file_extension="h5",
    ):
        """Process files in a specific subdirectory."""
        input_subdir = Path(input_subdir)
        output_subdir = Path(output_subdir)
        output_subdir.mkdir(parents=True, exist_ok=True)
        all_files = list(input_subdir.glob("**/*.npy"))
        print(f"Found {len(all_files)} files in {input_subdir}")

        for file_path in tqdm(
            all_files, desc=f"Processing {input_subdir.name}", ncols=80, leave=False
        ):
            rel = file_path.relative_to(input_subdir)
            out_file = output_subdir / rel.with_suffix(f".{file_extension}")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            self.process_file_to_h5(
                input_file=str(file_path),
                output_file=str(out_file),
                max_events=max_events,
                compression=compression,
            )
        print(f"✓ Done processing subdir: {input_subdir} → {output_subdir}")

    def save_model(self, path):
        """Save the trained model."""
        torch.save(self.encoder.state_dict(), path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path):
        """Load a pre-trained model."""
        self.encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.encoder.to(self.device)
        print(f"✓ Model loaded from {path}")

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.encoder.eval()

    def set_train_mode(self):
        """Set the model to training mode."""
        self.encoder.train()

    @staticmethod
    def load_features(file_path):
        """Load features from an H5 file."""
        with h5py.File(file_path, "r") as h5f:
            return h5f["features"][:]


def extract_event_features(
    input_dir,
    output_dir,
    model_dim=256,
    num_heads=8,
    num_layers=6,
    max_events=None,
    batch_size=512,
    use_compression=True,
    use_sub=False,
):
    """Extract features from preprocessed event data using transformer encoder.

    This function provides a high-level interface for feature extraction with
    automatic GPU detection and optimized processing.

    Args:
        input_dir: Directory containing preprocessed event data
        output_dir: Directory where feature vectors will be saved
        model_dim: Dimension of the transformer model
        num_heads: Number of attention heads in transformer
        num_layers: Number of transformer encoder layers
        max_events: Maximum number of events to process per file (for memory constraints)
        batch_size: Batch size for processing events
        use_compression: Whether to use compressed H5 format for outputs
        use_sub: Whether to process subdirectories individually

    Returns:
        None: Features are saved to the specified output directory

    """
    # Verify GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction. No GPU available.")

    device = "cuda"
    print("🚀 Starting feature extraction...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Device: {device}")
    print(f"   Model: {model_dim}D, {num_heads} heads, {num_layers} layers")
    print(f"   Batch size: {batch_size}")

    # Initialize the feature extractor
    extractor = OptimizedEventFeatureExtractor(
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        max_spatial_dim=(240, 180),
        device=device,
        batch_size=batch_size,
    )

    # Set processing parameters
    file_extension = "h5" if use_compression else "npy"
    compression = "gzip" if use_compression else None

    # Set model to evaluation mode
    extractor.set_eval_mode()

    # Validate input directory
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory '{input_path}' does not exist.")

    print(f"📁 Processing all event data files in {input_path}")

    # Process files
    if use_sub:
        extractor.process_subdirectory(
            input_subdir=input_path,
            output_subdir=output_path,
            max_events=max_events,
            compression=compression,
            file_extension=file_extension,
        )
    else:
        extractor.process_directory(
            input_dir=input_path,
            output_dir=output_path,
            max_events=max_events,
            compression=compression,
            file_extension=file_extension,
        )

    print("🎉 Feature extraction completed successfully!")
