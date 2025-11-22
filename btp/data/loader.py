from pathlib import Path


def load_datasets(download=True):
    """Imports datasets from Kaggle.

    Make sure you have kaggle credentials configured if download is True.
    """
    if not download:
        print("Skipping Kaggle download.")
        # Assuming data might be available locally or handled otherwise
        joymridha04_lle_vos_dataset_path = Path.cwd() / "working" / "lle-vos-dataset"
        joymridha04_preprocessed_lle_vos_dataset_path = (
            Path.cwd() / "working" / "preprocessed-lle-vos-dataset"
        )
        return (
            joymridha04_lle_vos_dataset_path,
            joymridha04_preprocessed_lle_vos_dataset_path,
        )

    try:
        import kagglehub
        # kagglehub.login() # This might require interactive login, which we want to avoid if possible or handle gracefully.
        # Assuming credentials are set up in the environment
    except ImportError:
        print("kagglehub not installed. Please install it to download datasets.")
        return None, None

    print("Importing datasets from Kaggle...")

    # Download LLE-VOS Dataset (Original raw data)
    try:
        joymridha04_lle_vos_dataset_path = kagglehub.dataset_download(
            "joymridha04/lle-vos-dataset"
        )
        print("✓ LLE-VOS Dataset imported successfully!")
        print(f"Dataset path: {joymridha04_lle_vos_dataset_path}")
    except Exception as e:
        print(f"Failed to download LLE-VOS Dataset: {e}")
        joymridha04_lle_vos_dataset_path = None

    # Download Preprocessed LLE-VOS Dataset
    try:
        joymridha04_preprocessed_lle_vos_dataset_path = kagglehub.dataset_download(
            "joymridha04/preprocessed-lle-vos-dataset"
        )
        print("✓ Preprocessed LLE-VOS Dataset imported successfully!")
        print(
            f"Preprocessed dataset path: {joymridha04_preprocessed_lle_vos_dataset_path}"
        )
    except Exception as e:
        print(f"Failed to download Preprocessed LLE-VOS Dataset: {e}")
        joymridha04_preprocessed_lle_vos_dataset_path = None

    return (
        joymridha04_lle_vos_dataset_path,
        joymridha04_preprocessed_lle_vos_dataset_path,
    )


def get_data_paths(raw_path=None, preprocessed_path=None):
    """Set up path variables for easy access."""
    raw_data_path = Path(raw_path) if raw_path else None
    preprocessed_data_path = Path(preprocessed_path) if preprocessed_path else None

    print("\nDataset structure:")
    if raw_data_path:
        print(f"Raw data: {raw_data_path}")
    if preprocessed_data_path:
        print(f"Preprocessed data: {preprocessed_data_path}")

    return raw_data_path, preprocessed_data_path
