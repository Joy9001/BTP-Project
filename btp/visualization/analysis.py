import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import h5py

def load_all_features(fused_features_dir, max_folders=70, max_files_per_folder=50):
    """Loads all fused feature vectors from the specified directory structure.

    Args:
        fused_features_dir (str): Path to the root directory of fused features.
        max_folders (int): The maximum number of folders to process.
        max_files_per_folder (int): The maximum number of files to load from each folder.

    Returns:
        tuple: A tuple containing:
            - all_features (np.ndarray): A 2D array of all loaded feature vectors.
            - labels (np.ndarray): An array of folder labels for each feature vector.
            - folder_map (dict): A dictionary mapping folder names to integer labels.

    """
    fused_dir = Path(fused_features_dir)
    all_features = []
    labels = []

    subdirectories = sorted([d for d in fused_dir.iterdir() if d.is_dir()])[
        :max_folders
    ]
    folder_map = {folder.name: i for i, folder in enumerate(subdirectories)}

    print(f"Loading features from {len(subdirectories)} folders...")
    for i, subdir in enumerate(tqdm(subdirectories, desc="Loading Folders")):
        files = sorted(list(subdir.glob("*.npy")))
        # Randomly sample files if there are too many, to keep visualization manageable
        if len(files) > max_files_per_folder:
            files = random.sample(files, max_files_per_folder)

        for file_path in files:
            try:
                feature = np.load(file_path)
                all_features.append(feature)
                labels.append(i)
            except Exception as e:
                print(f"Could not load {file_path}: {e}")

    return np.array(all_features), np.array(labels), folder_map

def visualize_tsne(
    features, labels, folder_map, title="t-SNE Visualization of Fused Features"
):
    """Performs t-SNE dimensionality reduction and visualizes the result.

    What this means:
    t-SNE is a technique for visualizing high-dimensional data in a 2D or 3D plot.
    It tries to place similar data points close to each other and dissimilar points
    far apart. This plot helps you see the overall structure of your dataset.
    If you see distinct clusters of colors, it means the ACMF module has learned
    to group data from the same original folder together, which is a sign of
    successful and meaningful feature fusion.
    """
    print("\nPerforming t-SNE... (This may take a few minutes)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(14, 10))
    # Use a color palette that has enough distinct colors
    palette = sns.color_palette("husl", len(folder_map))
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        s=50,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Folder", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_similarity_heatmap(
    features, labels, folder_map, title="Feature Similarity Heatmap"
):
    """Computes and displays a cosine similarity heatmap.

    What this means:
    This heatmap shows how similar each feature vector is to every other feature vector.
    The color of each square represents the cosine similarity (from 0, very dissimilar, to 1, identical).
    The features are ordered by their original folder. A perfect result would be bright yellow squares
    along the diagonal (high intra-folder similarity) and dark purple/blue squares everywhere else
    (low inter-folder similarity). This "blocky" pattern is strong evidence that your fusion
    process is creating distinct and consistent representations for each scene/folder.
    """
    print("\nCalculating similarity matrix for heatmap...")
    # Using a subset for performance if the dataset is too large
    if len(features) > 1000:
        print("Using a subset of 1000 features for heatmap visualization.")
        indices = np.random.choice(len(features), 1000, replace=False)
        features = features[indices]
        labels = labels[indices]

        # Sort by label to create the blocky pattern
        sort_order = np.argsort(labels)
        features = features[sort_order]
        labels = labels[sort_order]

    similarity_matrix = cosine_similarity(features)

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap="viridis")
    plt.title(title, fontsize=16)
    plt.xlabel("Feature Index", fontsize=12)
    plt.ylabel("Feature Index", fontsize=12)
    plt.show()

def visualize_single_feature_vector(feature, title="Single Fused Feature Vector"):
    """Displays a single feature vector as a bar chart.

    What this means:
    This is a direct look at the values that make up one feature vector. Each bar
    represents one of the 768 dimensions. This visualization is most useful for
    debugging. If you see a feature vector that is all zeros or has strange patterns,
    it might point to an issue with the specific image or event file it came from.
    It helps to confirm that the fusion process is producing non-trivial results.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature)), feature)
    plt.title(title, fontsize=16)
    plt.xlabel("Feature Dimension", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(axis="y")
    plt.show()

def visualize_before_vs_after(img_feat, evt_feat, fused_feat):
    """Compares original image, event, and final fused features.

    What this means:
    This visualization directly shows the effect of the fusion.
    - The top plot is the original 768-dim image feature.
    - The middle plot is the original 256-dim event feature.
    - The bottom plot is the final 768-dim fused feature.
    You can visually inspect how the characteristics of the two input vectors
    are combined to create the final output. The fused vector should look
    different from both inputs, demonstrating that the ACMF module is performing
    a complex, non-linear transformation rather than a simple addition or concatenation.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Image Feature
    axs[0].bar(range(len(img_feat)), img_feat, color="skyblue")
    axs[0].set_title("Original Image Feature (768-D)", fontsize=14)
    axs[0].set_ylabel("Value")
    axs[0].grid(axis="y")

    # Event Feature
    axs[1].bar(range(len(evt_feat)), evt_feat, color="salmon")
    axs[1].set_title("Original Event Feature (256-D)", fontsize=14)
    axs[1].set_ylabel("Value")
    axs[1].grid(axis="y")

    # Fused Feature
    axs[2].bar(range(len(fused_feat)), fused_feat, color="lightgreen")
    axs[2].set_title("Final Fused Feature (768-D)", fontsize=14)
    axs[2].set_xlabel("Feature Dimension")
    axs[2].set_ylabel("Value")
    axs[2].grid(axis="y")

    plt.tight_layout()
    plt.show()
