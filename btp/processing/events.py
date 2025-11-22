import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

class EventProcessor:
    """A class containing methods for processing raw event data.

    This class provides GPU-accelerated methods for:
    - Voxelizing raw event data
    - Applying adaptive low-pass filtering
    - Processing event data pipelines
    """

    @staticmethod
    def voxelize_events(events, voxel_size, time_window, normalize=True):
        """Voxelize raw event data with improved parameters using PyTorch on GPU.

        Args:
            events: Input event data (N, 4) - (x, y, t, p)
            voxel_size: Spatial voxel size (voxel_x, voxel_y)
            time_window: Temporal window for grouping events
            normalize: Whether to normalize timestamps

        Returns:
            torch.Tensor: Voxelized events (N, 4) - (voxel_x, voxel_y, time_bin, p)

        """
        # Convert to PyTorch tensor on GPU
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, device="cuda", dtype=torch.float32)

        # Extract event components
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

        # Get min timestamp for normalization
        t_min = t.min().item() if normalize else 0

        # Normalize spatial coordinates to voxel grid
        voxel_x = torch.floor(x / voxel_size[0]).to(torch.int)
        voxel_y = torch.floor(y / voxel_size[1]).to(torch.int)

        # Normalize temporal coordinates to time bins
        time_bins = torch.floor((t - t_min) / time_window).to(torch.int)

        # Combine into voxel representation
        voxelized_events = torch.stack((voxel_x, voxel_y, time_bins, p), dim=1)

        return voxelized_events

    @staticmethod
    def adaptive_low_pass_filter(events, density_threshold):
        """Apply adaptive low-pass filtering to voxelized event data using PyTorch on GPU.

        Args:
            events (torch.Tensor): Voxelized event data of shape (N, 4) where each row is (voxel_x, voxel_y, time_bin, p).
            density_threshold (int): Minimum number of events required in a voxel to retain it.

        Returns:
            torch.Tensor: Filtered event data.

        """
        # Extract xyz coordinates for uniqueness check
        voxel_coords = events[:, :3]

        # Use PyTorch's unique function
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_coords, dim=0, return_inverse=True, return_counts=True
        )

        # Find voxels that meet the density threshold
        dense_mask = counts >= density_threshold
        dense_voxel_indices = torch.where(dense_mask)[0]

        if len(dense_voxel_indices) == 0:
            # Return empty tensor with correct shape if no events pass the filter
            return torch.empty((0, 4), device="cuda", dtype=events.dtype)

        # Build mask for original events that belong to dense voxels
        event_mask = torch.zeros_like(inverse_indices, dtype=torch.bool)
        for idx in dense_voxel_indices:
            event_mask = event_mask | (inverse_indices == idx)

        # Filter events
        filtered_events = events[event_mask]

        return filtered_events

    @classmethod
    def process_event_data(cls, raw_events, voxel_size, time_window, density_threshold):
        """Process raw event data through voxelization and adaptive low-pass filtering with GPU.

        Args:
            raw_events (numpy.ndarray): Raw event data of shape (N, 4) where each row is (x, y, t, p).
            voxel_size (tuple): Spatial voxel size (voxel_x, voxel_y).
            time_window (float): Temporal window for grouping events.
            density_threshold (int): Minimum number of events required in a voxel to retain it.

        Returns:
            numpy.ndarray: Processed event data after voxelization and filtering.

        """
        # Step 1: Voxelize the raw events
        voxelized_events = cls.voxelize_events(raw_events, voxel_size, time_window)

        # Step 2: Apply adaptive low-pass filtering
        filtered_events = cls.adaptive_low_pass_filter(
            voxelized_events, density_threshold
        )

        # Convert back to numpy for saving
        filtered_events = filtered_events.cpu().numpy()

        return filtered_events


class BatchEventProcessor:
    """A class for batch processing of event data files with GPU acceleration.

    This class provides functionality to process multiple event data files using GPU
    acceleration, preserving directory structure and handling errors gracefully.
    """

    def __init__(self, voxel_size=(5, 5), time_window=1e3, density_threshold=2):
        """Initialize the batch processor with default parameters."""
        self.voxel_size = voxel_size
        self.time_window = time_window
        self.density_threshold = density_threshold

        # Check if GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for this implementation!")

    def process_all_event_files(self, input_dir, output_dir, batch_size=None):
        """Process all event data files in the input directory and save results to output directory using GPU.

        Args:
            input_dir (str or Path): Directory containing event data files.
            output_dir (str or Path): Directory to save processed files.
            batch_size (int, optional): Number of files to process in one GPU batch.

        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all .npy files in input directory and subdirectories
        all_files = list(input_dir.glob("**/*.npy"))
        print(f"Found {len(all_files)} files to process")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Process one file at a time by default, or use batch processing if specified
        if batch_size is None:
            for file_path in tqdm(all_files, desc="Processing event files with GPU"):
                # Maintain directory structure
                rel_path = file_path.relative_to(input_dir)
                output_file = output_dir / rel_path

                # Create output subdirectory
                output_file.parent.mkdir(parents=True, exist_ok=True)

                self._process_single_file_gpu(
                    (
                        file_path,
                        output_file,
                        self.voxel_size,
                        self.time_window,
                        self.density_threshold,
                    )
                )
        else:
            # TODO: Implement batch processing with GPU if needed
            raise NotImplementedError(
                "Batch processing with GPU is not yet implemented"
            )

        print(f"Processing complete. Results saved to {output_dir}")

    def _process_single_file_gpu(self, args):
        """Process a single event data file using GPU."""
        input_file, output_file, voxel_size, time_window, density_threshold = args

        try:
            # Load raw event data
            raw_event_data = np.load(input_file)

            # Process the data using GPU
            processed_events = EventProcessor.process_event_data(
                raw_event_data, voxel_size, time_window, density_threshold
            )

            # Save the processed data
            np.save(output_file, processed_events)

            return True
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return False
