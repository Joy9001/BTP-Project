from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class EventVisualizer:
    """A class for visualizing event data with various methods.

    This class provides methods for:
    - Loading event data
    - 2D scatter plots
    - Time surface visualization
    - Event density visualization
    """

    @staticmethod
    def load_event_data(file_path, use_gpu=False):
        """Load event data from a NumPy file.

        Parameters
        ----------
        file_path : str or Path
            Path to the .npy file containing event data
        use_gpu : bool
            Whether to load data to GPU (as PyTorch tensor)

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Event data with shape (N, 4) where each row is (x/voxel_x, y/voxel_y, t/time_bin, p)

        """
        event_data = np.load(file_path)

        if use_gpu and torch.cuda.is_available():
            return torch.tensor(event_data, device="cuda")

        return event_data

    @staticmethod
    def plot_events_2d(
        events,
        figsize=(10, 8),
        title="Event Visualization",
        color_pos="red",
        color_neg="blue",
        alpha=0.5,
        s=1,
    ):
        """Plot events in 2D space, distinguishing positive and negative events by color."""
        # Ensure we're working with numpy array
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Separate positive and negative events
        pos_events = events[events[:, 3] > 0]
        neg_events = events[events[:, 3] <= 0]

        # Plot positive events
        if len(pos_events) > 0:
            ax.scatter(
                pos_events[:, 0],
                pos_events[:, 1],
                c=color_pos,
                alpha=alpha,
                s=s,
                label="Positive",
            )

        # Plot negative events
        if len(neg_events) > 0:
            ax.scatter(
                neg_events[:, 0],
                neg_events[:, 1],
                c=color_neg,
                alpha=alpha,
                s=s,
                label="Negative",
            )

        # Invert y-axis to match image coordinates
        ax.invert_yaxis()

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        ax.legend()

        return fig, ax

    @staticmethod
    def plot_time_surface(
        events, width, height, time_bins=None, figsize=(12, 10), cmap="viridis"
    ):
        """Create a time surface visualization of events."""
        # Ensure we're working with numpy array
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()

        # Initialize time surface
        time_surface = np.zeros((height, width))

        # Normalize timestamps to [0, 1]
        if len(events) > 0:
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            t_norm = (
                (events[:, 2] - t_min) / (t_max - t_min)
                if t_max > t_min
                else events[:, 2]
            )

            # Create time surface (most recent events overwrite older ones)
            for x, y, t, p in zip(
                events[:, 0], events[:, 1], t_norm, events[:, 3], strict=False
            ):
                if 0 <= x < width and 0 <= y < height:
                    time_surface[int(y), int(x)] = t

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(time_surface, cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Normalized Time")
        ax.set_title("Time Surface Visualization")

        return fig, ax

    @staticmethod
    def visualize_event_density(events, width, height, figsize=(10, 8), cmap="hot"):
        """Visualize the density of events in 2D space."""
        # Ensure we're working with numpy array
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()

        # Initialize density map
        density_map = np.zeros((height, width))

        # Count events at each pixel location
        for x, y, _, _ in events:
            if 0 <= x < width and 0 <= y < height:
                density_map[int(y), int(x)] += 1

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            np.log1p(density_map), cmap=cmap, interpolation="nearest"
        )  # log1p for better visualization
        plt.colorbar(im, ax=ax, label="Log(Event Count + 1)")
        ax.set_title("Event Density Visualization")

        return fig, ax


# Add comparison visualization methods to EventVisualizer
class EventComparator(EventVisualizer):
    """Extended visualizer for comparing raw vs processed event data."""

    @classmethod
    def visualize_raw_vs_processed(cls, raw_file, processed_file, output_dir=None):
        """Visualize raw vs processed event data with multiple visualization methods.

        Parameters
        ----------
        raw_file : str or Path
            Path to the raw event data file
        processed_file : str or Path
            Path to the processed event data file
        output_dir : str or Path, optional
            Directory to save visualizations to

        """
        # Create output directory if provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load raw and processed event data
        raw_events = cls.load_event_data(raw_file)
        processed_events = cls.load_event_data(processed_file)

        print(f"Raw events: {raw_events.shape}")
        print(f"Processed events: {processed_events.shape}")

        # Get image dimensions from data
        if len(raw_events) > 0:
            width = int(raw_events[:, 0].max()) + 1
            height = int(raw_events[:, 1].max()) + 1
        else:
            width, height = 346, 260  # Default dimensions if no events

        # 1. 2D scatter plot visualization
        fig_raw, _ = cls.plot_events_2d(raw_events, title="Raw Events")
        if output_dir:
            fig_raw.savefig(
                output_dir / "raw_events_2d.png", dpi=300, bbox_inches="tight"
            )

        fig_proc, _ = cls.plot_events_2d(processed_events, title="Processed Events")
        if output_dir:
            fig_proc.savefig(
                output_dir / "processed_events_2d.png", dpi=300, bbox_inches="tight"
            )

        # 2. Time surface visualization
        fig_raw_ts, _ = cls.plot_time_surface(raw_events, width, height)
        if output_dir:
            fig_raw_ts.savefig(
                output_dir / "raw_time_surface.png", dpi=300, bbox_inches="tight"
            )

        fig_proc_ts, _ = cls.plot_time_surface(processed_events, width, height)
        if output_dir:
            fig_proc_ts.savefig(
                output_dir / "processed_time_surface.png", dpi=300, bbox_inches="tight"
            )

        # 3. Event density visualization
        fig_raw_den, _ = cls.visualize_event_density(raw_events, width, height)
        if output_dir:
            fig_raw_den.savefig(
                output_dir / "raw_density.png", dpi=300, bbox_inches="tight"
            )

        fig_proc_den, _ = cls.visualize_event_density(processed_events, width, height)
        if output_dir:
            fig_proc_den.savefig(
                output_dir / "processed_density.png", dpi=300, bbox_inches="tight"
            )

        # Show all figures if not saving
        if not output_dir:
            plt.show()
