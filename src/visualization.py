from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# SUIM CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "background",
    "human_diver",
    "aquatic_plants",
    "wrecks_ruins",
    "robots",
    "reefs_invertebrates",
    "fish_vertebrates",
    "sea_floor_rocks",
]


# ============================================================
# SUIM RGB COLORS
# ============================================================

CLASS_COLORS = {
    0: (0, 0, 0),          # Background
    1: (0, 0, 255),        # Human diver
    2: (0, 255, 0),        # Aquatic plants
    3: (0, 255, 255),      # Wrecks / ruins
    4: (255, 0, 0),        # Robots
    5: (255, 0, 255),      # Reefs / invertebrates
    6: (255, 255, 0),      # Fish / vertebrates
    7: (255, 255, 255),    # Sea-floor / rocks
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/vamsikrishnagondu/Desktop/Project 1"
)

TRAIN_IMAGES_DIR = (
    PROJECT_ROOT
    / "SUIM"
    / "train_val"
    / "images"
)

TRAIN_MASKS_DIR = (
    PROJECT_ROOT
    / "SUIM"
    / "train_val"
    / "masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load an underwater RGB image.
    """

    image = Image.open(image_path).convert("RGB")

    return np.array(image)


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):
    """
    Load a SUIM RGB segmentation mask.
    """

    mask = Image.open(mask_path).convert("RGB")

    return np.array(mask)


# ============================================================
# CONVERT CLASS MASK TO DISPLAYABLE RGB
# ============================================================

def class_mask_to_rgb(class_mask):
    """
    Convert a class-index mask into an RGB visualization.

    Input:
        class_mask -> [H, W]

    Output:
        RGB image -> [H, W, 3]
    """

    height, width = class_mask.shape

    rgb_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for class_id, color in CLASS_COLORS.items():

        rgb_mask[class_mask == class_id] = color

    return rgb_mask


# ============================================================
# RGB MASK → CLASS MASK
# ============================================================

def rgb_mask_to_class_mask(mask):
    """
    Convert SUIM RGB mask into integer class IDs.
    """

    class_mask = np.zeros(
        mask.shape[:2],
        dtype=np.uint8
    )

    for class_id, color in CLASS_COLORS.items():

        color_array = np.array(color)

        matches = np.all(
            mask == color_array,
            axis=-1
        )

        class_mask[matches] = class_id

    return class_mask


# ============================================================
# CREATE CLASS LEGEND
# ============================================================

def create_legend():

    from matplotlib.patches import Patch

    legend_items = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        color = np.array(
            CLASS_COLORS[class_id]
        ) / 255.0

        legend_items.append(
            Patch(
                facecolor=color,
                label=f"{class_id}: {class_name}"
            )
        )

    return legend_items


# ============================================================
# VISUALIZE ONE SAMPLE
# ============================================================

def visualize_sample(
    image_path,
    mask_path,
    save_path=None
):
    """
    Visualize:

        Original Image
        Ground Truth Mask
        Ground Truth Overlay

    """

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    image = load_image(image_path)

    rgb_mask = load_mask(mask_path)

    class_mask = rgb_mask_to_class_mask(
        rgb_mask
    )

    # --------------------------------------------------------
    # Convert class mask back to clean RGB
    # --------------------------------------------------------

    display_mask = class_mask_to_rgb(
        class_mask
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = (
        0.6 * image
        + 0.4 * display_mask
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(image)

    axes[0].set_title(
        "Original Underwater Image",
        fontsize=13
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(display_mask)

    axes[1].set_title(
        "Ground Truth Segmentation",
        fontsize=13
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[2].imshow(overlay)

    axes[2].set_title(
        "Ground Truth Overlay",
        fontsize=13
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_items = create_legend()

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(
            f"Saved visualization:\n{save_path}"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# FIND CORRESPONDING MASK
# ============================================================

def find_mask(image_path):

    image_path = Path(image_path)

    possible_masks = [
        TRAIN_MASKS_DIR
        / f"{image_path.stem}.bmp",

        TRAIN_MASKS_DIR
        / f"{image_path.stem}.png",

        TRAIN_MASKS_DIR
        / f"{image_path.stem}.jpg",
    ]

    for mask_path in possible_masks:

        if mask_path.exists():
            return mask_path

    return None


# ============================================================
# VISUALIZE FIRST SAMPLE
# ============================================================

def visualize_first_sample():

    image_files = sorted(
        [
            file
            for file in TRAIN_IMAGES_DIR.iterdir()
            if file.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        ]
    )

    if len(image_files) == 0:

        raise RuntimeError(
            f"No images found in:\n"
            f"{TRAIN_IMAGES_DIR}"
        )

    image_path = image_files[0]

    mask_path = find_mask(
        image_path
    )

    if mask_path is None:

        raise RuntimeError(
            f"No mask found for:\n"
            f"{image_path.name}"
        )

    output_path = (
        OUTPUT_DIR
        / "sample_ground_truth.png"
    )

    print("=" * 60)
    print("SUIM VISUALIZATION")
    print("=" * 60)

    print(
        f"\nImage: {image_path.name}"
    )

    print(
        f"Mask : {mask_path.name}"
    )

    visualize_sample(
        image_path,
        mask_path,
        output_path
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    visualize_first_sample()