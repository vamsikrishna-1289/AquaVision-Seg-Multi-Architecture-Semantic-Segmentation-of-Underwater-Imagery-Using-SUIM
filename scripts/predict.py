import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, random_split


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from src.dataset import SUIMDataset

from src.models.unet import create_unet
from src.models.deeplabv3plus import create_deeplabv3plus
from src.models.segformer import create_segformer


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 8

BATCH_SIZE = 1

NUM_WORKERS = 0

RANDOM_SEED = 42

NUM_SAMPLES = 10


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "SUIM",
    "train_val",
    "images"
)

MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "SUIM",
    "train_val",
    "masks"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints"
)

PREDICTION_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "predictions"
)

VISUALIZATION_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "visualizations"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "background",
    "human_diver",
    "aquatic_plants",
    "wrecks_ruins",
    "robots",
    "reefs_invertebrates",
    "fish_vertebrates",
    "sea_floor_rocks"
]


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    print(
        f"Loading checkpoint: "
        f"{os.path.basename(checkpoint_path)}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(model_name):

    if model_name == "U-Net":

        return create_unet(
            num_classes=NUM_CLASSES
        )

    elif model_name == "DeepLabV3+":

        return create_deeplabv3plus(
            num_classes=NUM_CLASSES
        )

    elif model_name == "SegFormer":

        return create_segformer(
            num_classes=NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# GET MODEL OUTPUT
# ============================================================

def get_model_output(output):

    if torch.is_tensor(output):

        return output

    if isinstance(output, (tuple, list)):

        return output[0]

    if isinstance(output, dict):

        if "out" in output:

            return output["out"]

        if "logits" in output:

            return output["logits"]

    raise TypeError(
        f"Unsupported model output type: "
        f"{type(output)}"
    )


# ============================================================
# MASK TO RGB
# ============================================================

def mask_to_rgb(mask):

    """
    Convert class-index mask into a visually
    distinguishable RGB image.
    """

    # --------------------------------------------------------
    # Fixed colors for 8 SUIM classes
    # --------------------------------------------------------

    colors = np.array([
        [0, 0, 0],          # background
        [0, 0, 255],        # human diver
        [0, 255, 0],        # aquatic plants
        [0, 255, 255],      # wrecks / ruins
        [255, 0, 0],        # robots
        [255, 0, 255],      # reefs / invertebrates
        [255, 255, 0],      # fish / vertebrates
        [255, 128, 0]       # sea floor / rocks
    ], dtype=np.uint8)

    rgb = colors[
        np.clip(
            mask,
            0,
            NUM_CLASSES - 1
        )
    ]

    return rgb


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

def load_original_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    return np.array(image)


# ============================================================
# GET ORIGINAL IMAGE PATH
# ============================================================

def get_image_path(
    dataset,
    dataset_index
):

    # --------------------------------------------------------
    # SUIMDataset stores image paths.
    #
    # Try the most common attribute names.
    # --------------------------------------------------------

    if hasattr(
        dataset,
        "image_paths"
    ):

        return dataset.image_paths[
            dataset_index
        ]

    if hasattr(
        dataset,
        "images"
    ):

        item = dataset.images[
            dataset_index
        ]

        if isinstance(
            item,
            str
        ):

            return item

    if hasattr(
        dataset,
        "image_files"
    ):

        return os.path.join(
            IMAGE_DIR,
            dataset.image_files[
                dataset_index
            ]
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    files = sorted([
        os.path.join(
            IMAGE_DIR,
            filename
        )
        for filename in os.listdir(
            IMAGE_DIR
        )
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    return files[dataset_index]


# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

def create_comparison_figure(
    original,
    ground_truth,
    predictions,
    filename
):

    model_names = [
        "U-Net",
        "DeepLabV3+",
        "SegFormer"
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        original
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(
        mask_to_rgb(
            ground_truth
        )
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Model predictions
    # --------------------------------------------------------

    for index, model_name in enumerate(
        model_names
    ):

        prediction = predictions[
            model_name
        ]

        axes[index + 2].imshow(
            mask_to_rgb(
                prediction
            )
        )

        axes[index + 2].set_title(
            model_name
        )

        axes[index + 2].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"SUIM Semantic Segmentation Comparison\n{filename}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = os.path.join(
        VISUALIZATION_DIR,
        f"{os.path.splitext(filename)[0]}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original,
    prediction,
    filename,
    model_name
):

    mask_rgb = mask_to_rgb(
        prediction
    )

    # --------------------------------------------------------
    # Resize mask if necessary
    # --------------------------------------------------------

    if mask_rgb.shape[:2] != original.shape[:2]:

        mask_rgb = np.array(
            Image.fromarray(
                mask_rgb
            ).resize(
                (
                    original.shape[1],
                    original.shape[0]
                ),
                Image.Resampling.NEAREST
            )
        )

    # --------------------------------------------------------
    # Blend
    # --------------------------------------------------------

    overlay = (
        0.65 * original.astype(
            np.float32
        )
        +
        0.35 * mask_rgb.astype(
            np.float32
        )
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    output_filename = (
        f"{os.path.splitext(filename)[0]}"
        f"_{model_name.lower().replace('+', 'plus').replace('-', '')}"
        f"_overlay.png"
    )

    output_path = os.path.join(
        VISUALIZATION_DIR,
        output_filename
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )

    return output_path


# ============================================================
# SAVE PREDICTION MASK
# ============================================================

def save_prediction_mask(
    prediction,
    filename,
    model_name
):

    model_folder = os.path.join(
        PREDICTION_DIR,
        model_name.lower()
        .replace("+", "plus")
        .replace("-", "")
    )

    os.makedirs(
        model_folder,
        exist_ok=True
    )

    output_filename = (
        f"{os.path.splitext(filename)[0]}"
        f"_prediction.png"
    )

    output_path = os.path.join(
        model_folder,
        output_filename
    )

    # --------------------------------------------------------
    # Save class-index mask
    #
    # PNG stores the class IDs directly.
    # --------------------------------------------------------

    Image.fromarray(
        prediction.astype(
            np.uint8
        )
    ).save(
        output_path
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SUIM MODEL PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True
    )

    os.makedirs(
        VISUALIZATION_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading dataset..."
    )

    dataset = SUIMDataset(
        IMAGE_DIR,
        MASK_DIR,
        training=False
    )

    total_samples = len(dataset)

    # --------------------------------------------------------
    # Same validation split used during training/evaluation
    # --------------------------------------------------------

    validation_size = int(
        0.2 * total_samples
    )

    training_size = (
        total_samples
        - validation_size
    )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    _, validation_dataset = random_split(
        dataset,
        [
            training_size,
            validation_size
        ],
        generator=generator
    )

    print(
        f"Total samples     : {total_samples}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    model_configs = [
        (
            "U-Net",
            "unet_best.pth"
        ),
        (
            "DeepLabV3+",
            "deeplabv3plus_best.pth"
        ),
        (
            "SegFormer",
            "segformer_best.pth"
        )
    ]

    models = {}

    # ========================================================
    # LOAD ALL MODELS
    # ========================================================

    for model_name, checkpoint_name in model_configs:

        print()
        print(
            f"Loading {model_name}..."
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            checkpoint_name
        )

        if not os.path.exists(
            checkpoint_path
        ):

            print(
                f"WARNING: checkpoint not found:"
            )

            print(
                checkpoint_path
            )

            continue

        model = create_model(
            model_name
        )

        model = model.to(
            device
        )

        model = load_checkpoint(
            model,
            checkpoint_path,
            device
        )

        model.eval()

        models[
            model_name
        ] = model

        print(
            f"{model_name} loaded successfully."
        )

    # --------------------------------------------------------
    # Make sure models exist
    # --------------------------------------------------------

    if len(models) == 0:

        raise RuntimeError(
            "No trained model checkpoints were found."
        )

    # ========================================================
    # SELECT SAMPLES
    # ========================================================

    num_samples = min(
        NUM_SAMPLES,
        len(validation_dataset)
    )

    print()
    print(
        f"Generating predictions for "
        f"{num_samples} validation images..."
    )

    # ========================================================
    # PREDICTION LOOP
    # ========================================================

    with torch.no_grad():

        for sample_number in range(
            num_samples
        ):

            # ------------------------------------------------
            # Get sample
            # ------------------------------------------------

            image_tensor, ground_truth = (
                validation_dataset[
                    sample_number
                ]
            )

            # ------------------------------------------------
            # Add batch dimension
            # ------------------------------------------------

            input_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            ground_truth_np = (
                ground_truth
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Get original image path
            # ------------------------------------------------

            # random_split returns Subset,
            # so access the underlying dataset.

            original_dataset = (
                validation_dataset.dataset
            )

            original_index = (
                validation_dataset.indices[
                    sample_number
                ]
            )

            image_path = get_image_path(
                original_dataset,
                original_index
            )

            filename = os.path.basename(
                image_path
            )

            # ------------------------------------------------
            # Original image
            # ------------------------------------------------

            original_image = load_original_image(
                image_path
            )

            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            predictions = {}

            # ------------------------------------------------
            # Run all models
            # ------------------------------------------------

            for model_name, model in models.items():

                output = model(
                    input_tensor
                )

                output = get_model_output(
                    output
                )

                prediction = torch.argmax(
                    output,
                    dim=1
                )

                prediction = (
                    prediction[0]
                    .cpu()
                    .numpy()
                )

                predictions[
                    model_name
                ] = prediction

                # ------------------------------------------------
                # Save prediction mask
                # ------------------------------------------------

                prediction_path = (
                    save_prediction_mask(
                        prediction,
                        filename,
                        model_name
                    )
                )

                print(
                    f"[{sample_number + 1}/"
                    f"{num_samples}] "
                    f"{model_name}: "
                    f"{os.path.basename(prediction_path)}"
                )

                # ------------------------------------------------
                # Save overlay
                # ------------------------------------------------

                create_overlay(
                    original_image,
                    prediction,
                    filename,
                    model_name
                )

            # ------------------------------------------------
            # Comparison figure
            # ------------------------------------------------

            comparison_path = (
                create_comparison_figure(
                    original_image,
                    ground_truth_np,
                    predictions,
                    filename
                )
            )

            print(
                f"Comparison saved: "
                f"{comparison_path}"
            )

            print("-" * 60)

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("PREDICTION GENERATION COMPLETE")
    print("=" * 60)

    print()
    print(
        "Prediction masks:"
    )

    print(
        PREDICTION_DIR
    )

    print()
    print(
        "Visualization files:"
    )

    print(
        VISUALIZATION_DIR
    )

    print()
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()