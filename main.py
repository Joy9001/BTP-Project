import random
from pathlib import Path

# Import modules
from btp.data.loader import load_datasets
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


def main():
    # 1. Setup and Data Import
    print("--- 1. Setup and Data Import ---")
    # For local testing, we might not want to download every time or if credentials are missing.
    # Adjust download=True/False as needed or based on env var.
    raw_data_path, preprocessed_data_path = load_datasets(download=False)

    # Define paths (using local paths if kaggle paths are not available)
    # If running in the original environment, these paths might need to be adjusted to match
    # where kagglehub puts things, or we use the return values from load_datasets.

    # Default paths mirroring the original script logic but adaptable
    base_dir = Path.cwd()

    # If load_datasets returned None, we assume data is at specific locations or we can't proceed with some steps
    if raw_data_path is None:
        raw_data_path = base_dir / "data" / "raw"  # Placeholder
        print(f"Using placeholder raw data path: {raw_data_path}")
    else:
        raw_data_path = Path(raw_data_path)

    if preprocessed_data_path is None:
        preprocessed_data_path = base_dir / "data" / "preprocessed"  # Placeholder
        print(f"Using placeholder preprocessed data path: {preprocessed_data_path}")
    else:
        preprocessed_data_path = Path(preprocessed_data_path)

    working_dir = base_dir / "working"
    working_dir.mkdir(exist_ok=True)

    # 2. Event Data Preprocessing
    print("\n--- 2. Event Data Preprocessing ---")
    event_process_input = raw_data_path / "Dataset" / "Lowlight_event"
    event_process_output = (
        working_dir / "Preprocessed-Dataset" / "Processed_Lowlight_event"
    )

    if event_process_input.exists():
        processor = BatchEventProcessor(
            voxel_size=(5, 5), time_window=1e3, density_threshold=2
        )
        processor.process_all_event_files(
            input_dir=event_process_input, output_dir=event_process_output
        )
    else:
        print(
            f"Skipping event preprocessing: Input directory {event_process_input} does not exist."
        )

    # 3. Image Data Preprocessing
    print("\n--- 3. Image Data Preprocessing ---")
    input_image_process_path = raw_data_path / "Dataset" / "Lowlight_Images"
    output_image_process_path = (
        working_dir / "Preprocessed-Dataset" / "Processed_Lowlight_Images"
    )

    if input_image_process_path.exists():
        image_processor = ImageProcessor()
        image_processor.process_all_images(
            input_image_process_path, output_image_process_path
        )
    else:
        print(
            f"Skipping image preprocessing: Input directory {input_image_process_path} does not exist."
        )

    # 4. Visualization
    print("\n--- 4. Visualization ---")
    # Event Visualization
    # Need to find a sample file.
    if event_process_output.exists():
        sample_files = list(event_process_output.glob("**/*.npy"))
        if sample_files:
            processed_file = sample_files[0]
            # Assuming raw file has same relative path structure if we want comparison
            # But for now let's just try to visualize if we have the data
            # Construct raw file path if possible, otherwise skip comparison
            try:
                rel_path = processed_file.relative_to(event_process_output)
                raw_file = event_process_input / rel_path
                if raw_file.exists():
                    output_viz_dir = working_dir / "visualize" / "event_data"
                    EventComparator.visualize_raw_vs_processed(
                        raw_file, processed_file, output_dir=output_viz_dir
                    )
            except Exception as e:
                print(f"Could not run event comparison: {e}")

    # Image Visualization
    if input_image_process_path.exists() and output_image_process_path.exists():
        save_dir = working_dir / "visualize" / "image_data"
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

    # 5. Feature Extraction
    print("\n--- 5. Feature Extraction ---")

    # Event Feature Extraction
    extract_event_output_dir_base = (
        working_dir / "Extracted-Dataset" / "Extracted_Lowlight_event"
    )

    # In the original script, it iterates over directories 00001 to 00070.
    # We can replicate that or just process what we have.
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
    extract_image_output_dir = (
        working_dir / "Extracted-Dataset" / "Extracted_Lowlight_Images"
    )

    # Use preprocessed data path if available, or the one we just generated
    # The original script uses preprocessed_data_path variable which comes from kaggle download
    # If we ran preprocessing, we should use output_image_process_path

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

    # 6. Fusion
    print("\n--- 6. Fusion ---")
    fused_output_root = working_dir / "Extracted-Dataset" / "Fused_Features"

    # Inputs for fusion
    img_feat_root = extract_image_output_dir
    evt_feat_root = extract_event_output_dir_base

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

    # 7. Analysis Visualization
    print("\n--- 7. Analysis Visualization ---")
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

                # Before vs After comparison needs original features.
                # This is complex to orchestrate without knowing exact file matches in this generic script.
                # Skipping for now or could be added if we track file paths carefully.

            except Exception as e:
                print(f"Error in analysis visualization: {e}")
    else:
        print("Skipping analysis: Fused features not found.")


if __name__ == "__main__":
    main()
