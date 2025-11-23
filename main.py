import argparse
from pathlib import Path

from btp.features.events import check_feature_extraction
from btp.features.images import check_image_features
from btp.processing.events import BatchEventProcessor
from btp.processing.images import ImageProcessor


def setup_context():
    """Sets up the paths for local development."""
    print("--- 1. Setup and Data Import ---")

    base_dir = Path.cwd()

    # Define clean local data paths
    # User should place raw data in: project_root/data/raw
    raw_data_path = base_dir / "data" / "raw"

    # Outputs will go to: project_root/data/processed and project_root/data/features
    processed_dir = base_dir / "data" / "processed"
    features_dir = base_dir / "data" / "features"
    viz_dir = base_dir / "visualizations"

    # Create necessary directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "base_dir": base_dir,
        "raw_data_path": raw_data_path,
        "working_dir": base_dir,  # Keeping for compatibility if needed
        # INPUTS (Where your raw data sits)
        # Expects: data/raw/Dataset/Lowlight_event
        "event_process_input": raw_data_path / "Dataset" / "Lowlight_event",
        # Expects: data/raw/Dataset/Lowlight_Images
        "image_process_input": raw_data_path / "Dataset" / "Lowlight_Images",
        # INTERMEDIATE (Where voxel grids and denoised images go)
        "event_process_output": processed_dir / "Processed_Lowlight_event",
        "image_process_output": processed_dir / "Processed_Lowlight_Images",
        # FEATURES (Where .npy and .h5 embeddings go)
        "extract_event_output_dir_base": features_dir / "Extracted_Lowlight_event",
        "extract_image_output_dir": features_dir / "Extracted_Lowlight_Images",
        "fused_output_root": features_dir / "Fused_Features",
        # VISUALIZATION
        "viz_dir": viz_dir,
    }

    # Validation print
    print(f"📂 Expecting Raw Events at: {context['event_process_input']}")
    print(f"📂 Expecting Raw Images at: {context['image_process_input']}")

    return context


def run_preprocess_events(context):
    print("\n--- 2a. Event Data Preprocessing ---")
    event_process_input = context["event_process_input"]
    event_process_output = context["event_process_output"]

    if event_process_input.exists():
        # UPDATES:
        # 1. Removed voxel_size (now hardcoded to 260x346 in the class)
        # 2. Changed time_window to 50000 (50ms) because your data is in microseconds.
        #    1e3 (1ms) was too short for object detection.
        processor = BatchEventProcessor(time_window=50000, density_threshold=2)

        processor.process_all_event_files(
            input_dir=event_process_input, output_dir=event_process_output
        )
    else:
        print(
            f"Skipping event preprocessing: Input directory {event_process_input} does not exist."
        )


def run_preprocess_images(context):
    print("\n--- 2b. Image Data Preprocessing ---")
    image_process_input = context["image_process_input"]
    image_process_output = context["image_process_output"]

    if image_process_input.exists():
        processor = ImageProcessor()
        processor.process_all_images(
            input_dir=image_process_input, output_dir=image_process_output
        )
    else:
        print(
            f"Skipping image preprocessing: Input directory {image_process_input} does not exist."
        )


def check_event_feature_extraction(context):
    print("\n--- 3a. Event Feature Extraction (Verification) ---")
    # We verify the model works, but we don't save features to disk
    # because they are computed live during training.
    event_process_output = context["event_process_output"]

    if event_process_output.exists():
        check_feature_extraction(event_process_output)
    else:
        print("Skipping verification: No processed event data found.")


def check_image_feature_extraction(context):
    # ... (Existing Event check) ...

    print("\n--- 3b. Image Feature Extraction (Verification) ---")
    image_process_output = context["image_process_output"]
    if image_process_output.exists():
        check_image_features(image_process_output)


def main():
    parser = argparse.ArgumentParser(description="BTP Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=[
            "all",
            "setup",
            "preprocess_events",
            "preprocess_images",
            "check_event_features_extraction",
            "check_image_features_extraction",
        ],
        help="Step to run",
    )

    args = parser.parse_args()

    context = setup_context()
    steps = {
        "preprocess_events": run_preprocess_events,
        "preprocess_images": run_preprocess_images,
        "check_event_features_extraction": check_event_feature_extraction,
        "check_image_features_extraction": check_image_feature_extraction,
    }

    if args.step == "all":
        for _, step_func in steps.items():
            step_func(context)
    elif args.step == "setup":
        pass  # setup_context is always called
    elif args.step in steps:
        steps[args.step](context)


if __name__ == "__main__":
    main()
