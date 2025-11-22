import argparse
import random
from pathlib import Path

# Import modules
from btp.features.events import extract_event_features
from btp.features.images import ImageFeatureExtractor
from btp.fusion.utils import fuse_features_in_directories
from btp.processing.events import BatchEventProcessor
from btp.processing.images import ImageProcessor
from btp.visualization.analysis import (
    load_all_features,
    visualize_similarity_heatmap,
    visualize_single_feature_vector,
    visualize_tsne,
)
from btp.visualization.events import EventComparator
from btp.visualization.images import ImageVisualizer


def setup_context(download=False):
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
        "input_image_process_path": raw_data_path / "Dataset" / "Lowlight_Images",
        # INTERMEDIATE (Where voxel grids and denoised images go)
        "event_process_output": processed_dir / "Processed_Lowlight_event",
        "output_image_process_path": processed_dir / "Processed_Lowlight_Images",
        # FEATURES (Where .npy and .h5 embeddings go)
        "extract_event_output_dir_base": features_dir / "Extracted_Lowlight_event",
        "extract_image_output_dir": features_dir / "Extracted_Lowlight_Images",
        "fused_output_root": features_dir / "Fused_Features",
        # VISUALIZATION
        "viz_dir": viz_dir,
    }

    # Validation print
    print(f"📂 Expecting Raw Events at: {context['event_process_input']}")
    print(f"📂 Expecting Raw Images at: {context['input_image_process_path']}")

    return context


def run_preprocess_events(context):
    print("\n--- 2. Event Data Preprocessing ---")
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
    print("\n--- 3. Image Data Preprocessing ---")
    input_image_process_path = context["input_image_process_path"]
    output_image_process_path = context["output_image_process_path"]

    if input_image_process_path.exists():
        image_processor = ImageProcessor()
        image_processor.process_all_images(
            input_image_process_path, output_image_process_path
        )
    else:
        print(
            f"Skipping image preprocessing: Input directory {input_image_process_path} does not exist."
        )


def run_visualization(context):
    print("\n--- 4. Visualization ---")
    event_process_output = context["event_process_output"]
    event_process_input = context["event_process_input"]
    input_image_process_path = context["input_image_process_path"]
    output_image_process_path = context["output_image_process_path"]

    # Event Visualization
    if event_process_output.exists():
        sample_files = list(event_process_output.glob("**/*.npy"))
        if sample_files:
            processed_file = sample_files[0]
            # Assuming raw file has same relative path structure if we want comparison
            try:
                rel_path = processed_file.relative_to(event_process_output)
                raw_file = event_process_input / rel_path
                if raw_file.exists():
                    output_viz_dir = context["viz_dir"] / "event_data"
                    EventComparator.visualize_raw_vs_processed(
                        raw_file, processed_file, output_dir=output_viz_dir
                    )
            except Exception as e:
                print(f"Could not run event comparison: {e}")

    # Image Visualization
    if input_image_process_path.exists() and output_image_process_path.exists():
        save_dir = context["viz_dir"] / "image_data"
        try:
            ImageVisualizer.visualize_preprocessing(
                input_image_process_path,
                output_image_process_path,
                num_examples=5,
                save_dir=save_dir,
            )
            ImageVisualizer.plot_histograms(
                input_image_process_path,
                output_image_process_path,
                num_examples=5,
                save_dir=save_dir,
            )
        except Exception as e:
            print(f"Could not run image visualization: {e}")


def run_feature_extraction(context):
    print("\n--- 5. Feature Extraction ---")
    # working_dir = context["working_dir"]
    event_process_output = context["event_process_output"]
    extract_event_output_dir_base = context["extract_event_output_dir_base"]
    output_image_process_path = context["output_image_process_path"]
    preprocessed_data_path = context["preprocessed_data_path"]
    extract_image_output_dir = context["extract_image_output_dir"]

    # Event Feature Extraction
    # In the original script, it iterates over directories 00001 to 00070.
    if event_process_output.exists():
        # processing all subdirectories found
        for subdir in event_process_output.iterdir():
            if subdir.is_dir():
                print(f"Processing event directory: {subdir.name}")
                extract_output_dir = extract_event_output_dir_base / subdir.name
                try:
                    extract_event_features(
                        input_dir=str(subdir),
                        output_dir=str(extract_output_dir),
                        model_dim=256,
                        num_heads=8,
                        num_layers=6,
                        batch_size=512,
                        use_compression=True,
                        use_sub=False,  # input_dir is already the subdir
                    )
                except Exception as e:
                    print(f"Error extracting event features for {subdir.name}: {e}")

    # Image Feature Extraction
    image_feat_input_dir = output_image_process_path
    if not image_feat_input_dir.exists() and preprocessed_data_path:
        image_feat_input_dir = (
            preprocessed_data_path
            / "Preprocessed-Dataset"
            / "Processed_Lowlight_Images"
        )

    if image_feat_input_dir.exists():
        image_feature_extractor = ImageFeatureExtractor(
            model_name="google/vit-base-patch16-224-in21k", batch_size=16
        )
        try:
            image_feature_extractor.process_directory(
                input_dir=str(image_feat_input_dir),
                output_dir=str(extract_image_output_dir),
            )
        except Exception as e:
            print(f"Error extracting image features: {e}")
    else:
        print("Skipping image feature extraction: Input directory not found.")


def run_fusion(context):
    print("\n--- 6. Fusion ---")
    fused_output_root = context["fused_output_root"]
    img_feat_root = context["extract_image_output_dir"]
    evt_feat_root = context["extract_event_output_dir_base"]

    if img_feat_root.exists() and evt_feat_root.exists():
        fuse_features_in_directories(
            image_features_dir=str(img_feat_root),
            event_features_dir=str(evt_feat_root),
            output_dir=str(fused_output_root),
            image_feature_dim=768,
            event_feature_dim=256,
        )
    else:
        print("Skipping fusion: Feature directories not found.")


def run_analysis(context):
    print("\n--- 7. Analysis Visualization ---")
    fused_output_root = context["fused_output_root"]

    if fused_output_root.exists():
        features, labels, folder_map = load_all_features(
            str(fused_output_root), max_folders=10
        )
        if features.size > 0:
            try:
                visualize_tsne(features, labels, folder_map)
                visualize_similarity_heatmap(features, labels, folder_map)

                if len(features) > 0:
                    random_index = random.randint(0, len(features) - 1)
                    visualize_single_feature_vector(features[random_index])

            except Exception as e:
                print(f"Error in analysis visualization: {e}")
    else:
        print("Skipping analysis: Fused features not found.")


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
            "visualize",
            "extract_features",
            "fuse",
            "analyze",
        ],
        help="Step to run",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download datasets from Kaggle"
    )

    args = parser.parse_args()

    context = setup_context(download=args.download)

    steps = {
        "preprocess_events": run_preprocess_events,
        "preprocess_images": run_preprocess_images,
        "visualize": run_visualization,
        "extract_features": run_feature_extraction,
        "fuse": run_fusion,
        "analyze": run_analysis,
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
