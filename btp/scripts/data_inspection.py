from pathlib import Path

import numpy as np

# Replace with path to ONE of your .npy files
# file_path = (
#     Path.cwd() / "data" / "raw" / "Dataset" / "Lowlight_event" / "00001" / "00001.npy"
# )

file_path = (
    Path.cwd()
    / "data"
    / "raw"
    / "Dataset"
    / "Processed_Lowlight_event"
    / "00001"
    / "00001.npy"
)

try:
    events = np.load(file_path)
    print(f"--- File: {file_path} ---")
    print(f"Shape: {events.shape}")
    print(f"Data Type: {events.dtype}")

    # Assuming columns are (x, y, t, p) based on your previous code
    x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    print("\n--- Statistics ---")
    print(f"X range: min={x.min()}, max={x.max()}")
    print(f"Y range: min={y.min()}, max={y.max()}")
    print(f"T range: min={t.min()}, max={t.max()}")
    print(f"P values: {np.unique(p)}")

    # Guessing Timestamp Unit
    duration = t.max() - t.min()
    print("\n--- Duration Guess ---")
    print(f"Total Duration (raw units): {duration:.0f}")
    if duration > 1e5:
        print(" -> Likely MICROSECONDS (us). 1 sec = 1,000,000 units")
    else:
        print(" -> Likely SECONDS (s) or MILLISECONDS (ms)")

except Exception as e:
    print(f"Error inspecting file: {e}")
