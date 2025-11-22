from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm  # Fixed import


class EventProcessor:
    """Processing pipeline for DAVIS346 Event Data in Low-Light Conditions."""

    # DAVIS346 Resolution (Height, Width)
    SENSOR_SHAPE = (260, 346)

    @staticmethod
    def to_device(events, device="cuda"):
        if not isinstance(events, torch.Tensor):
            return torch.tensor(events, device=device, dtype=torch.float32)
        return events.to(device)

    @staticmethod
    def hot_pixel_filter(events, hot_threshold=100):
        """Removes pixels that fire too frequently."""
        coords = events[:, :2].long()
        unique_coords, inverse, counts = torch.unique(
            coords, dim=0, return_inverse=True, return_counts=True
        )
        is_hot_pixel = counts > hot_threshold
        valid_mask = ~is_hot_pixel[inverse]
        return events[valid_mask]

    @staticmethod
    def filter_by_voxel_density(events, min_density=2):
        """Retains events only if they fall into a voxel with at least `min_density` events."""
        x_coarse = (events[:, 0] / 2).long()
        y_coarse = (events[:, 1] / 2).long()
        t_coarse = (events[:, 2] / 50000).long()

        coarse_coords = torch.stack([x_coarse, y_coarse, t_coarse], dim=1)
        _, inverse, counts = torch.unique(
            coarse_coords, dim=0, return_inverse=True, return_counts=True
        )
        is_dense = counts >= min_density
        valid_mask = is_dense[inverse]
        return events[valid_mask]

    @staticmethod
    def voxelize_to_tensor(events, time_bin_size_us, num_bins=None):
        """Converts raw events to a Dense Voxel Grid Tensor (T, 2, H, W)."""
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        H, W = EventProcessor.SENSOR_SHAPE

        # 1. Normalize Time
        t_start = t.min()
        t_rel = t - t_start
        t_idx = torch.floor(t_rel / time_bin_size_us).long()

        if num_bins is not None:
            mask = t_idx < num_bins
            x, y, t_idx, p = x[mask], y[mask], t_idx[mask], p[mask]
        else:
            num_bins = int(t_idx.max().item()) + 1

        # 2. Quantize Space
        x_idx = torch.clamp(torch.floor(x).long(), 0, W - 1)
        y_idx = torch.clamp(torch.floor(y).long(), 0, H - 1)

        # 3. Handle Polarity
        p_idx = p.long()

        # 4. Create Dense Tensor
        dense_grid = torch.zeros(
            (num_bins, 2, H, W), device=events.device, dtype=torch.float32
        )

        flat_indices = (t_idx * 2 * H * W) + (p_idx * H * W) + (y_idx * W) + x_idx
        values = torch.ones_like(flat_indices, dtype=torch.float32)
        dense_grid.view(-1).scatter_add_(0, flat_indices, values)

        return dense_grid

    @classmethod
    def process_file(
        cls, events_np, time_window=50000, density_threshold=2, fixed_bins=4
    ):
        """Full pipeline: Load -> GPU -> Denoise -> Voxelize with FIXED temporal size for Batching.

        fixed_bins=4 ensures output is always (4, 2, 260, 346) for 200ms clips.
        """
        # 1. Load to GPU
        events = cls.to_device(events_np)

        # 2. Low-Light Denoising
        events = cls.hot_pixel_filter(events, hot_threshold=50)
        events = cls.filter_by_voxel_density(events, min_density=density_threshold)

        # 3. Voxelize (Enforce fixed size)
        if len(events) == 0:
            return torch.zeros(
                (fixed_bins, 2, cls.SENSOR_SHAPE[0], cls.SENSOR_SHAPE[1]),
                device=events.device,
            )

        # Pass num_bins to force the output shape
        tensor_grid = cls.voxelize_to_tensor(
            events, time_bin_size_us=time_window, num_bins=fixed_bins
        )

        return tensor_grid


class BatchEventProcessor:
    """Batch processor for multiple event files using GPU acceleration."""

    def __init__(self, time_window=50000, density_threshold=2):
        """time_window: in microseconds (e.g. 50000 = 50ms)"""
        self.time_window = time_window
        self.density_threshold = density_threshold

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for this implementation!")

    def process_all_event_files(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_files = list(input_dir.glob("**/*.npy"))
        print(
            f"Found {len(all_files)} files. Processing on {torch.cuda.get_device_name(0)}..."
        )

        for file_path in tqdm(all_files, desc="Processing"):
            rel_path = file_path.relative_to(input_dir)
            output_file = output_dir / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            self._process_single_file_gpu(file_path, output_file)

    def _process_single_file_gpu(self, input_file, output_file):
        try:
            raw_event_data = np.load(input_file)

            # FIXED: Call the correct method name and pass parameters
            processed_tensor = EventProcessor.process_file(
                raw_event_data,
                time_window=self.time_window,
                density_threshold=self.density_threshold,
            )

            # FIXED: Move to CPU and convert to numpy before saving
            # processed_tensor is (T, 2, H, W)
            np.save(output_file, processed_tensor.cpu().numpy())

            return True
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return False
