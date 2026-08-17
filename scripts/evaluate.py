import os
import sys
import csv
import time

import torch
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
# PROJECT IMPORTS
# ============================================================

from src.dataset import SUIMDataset

from src.models.unet import create_unet
from src.models.deeplabv3plus import create_deeplabv3plus
from src.models.segformer import create_segformer


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 8

IMAGE_SIZE = 256

BATCH_SIZE = 4

NUM_WORKERS = 0

RANDOM_SEED = 42

DATASET_IMAGES = os.path.join(
    PROJECT_ROOT,
    "SUIM",
    "train_val",
    "images"
)

DATASET_MASKS = os.path.join(
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

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)


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
# MODEL CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model, checkpoint_path, device):

    print(f"Loading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Different possible checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Assume checkpoint itself is state_dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove possible "module." prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load model weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# MODEL OUTPUT HANDLER
# ============================================================

def get_model_output(output):

    # Some models may return a tensor directly.
    if torch.is_tensor(output):

        return output

    # Handle tuple/list outputs.
    if isinstance(output, (tuple, list)):

        return output[0]

    # Handle dictionary outputs.
    if isinstance(output, dict):

        if "out" in output:
            return output["out"]

        if "logits" in output:
            return output["logits"]

    raise TypeError(
        "Unsupported model output type: "
        f"{type(output)}"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets,
    num_classes=NUM_CLASSES
):

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    pixel_accuracy = (
        (predictions == targets)
        .float()
        .mean()
        .item()
    )

    iou_scores = []
    dice_scores = []

    for class_id in range(num_classes):

        pred_class = predictions == class_id
        target_class = targets == class_id

        intersection = (
            pred_class & target_class
        ).sum().item()

        prediction_pixels = pred_class.sum().item()
        target_pixels = target_class.sum().item()

        union = (
            prediction_pixels
            + target_pixels
            - intersection
        )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        if union == 0:

            iou = 1.0

        else:

            iou = intersection / union

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        denominator = (
            prediction_pixels
            + target_pixels
        )

        if denominator == 0:

            dice = 1.0

        else:

            dice = (
                2.0 * intersection
            ) / denominator

        iou_scores.append(iou)
        dice_scores.append(dice)

    mean_iou = sum(iou_scores) / num_classes
    mean_dice = sum(dice_scores) / num_classes

    return (
        pixel_accuracy,
        mean_iou,
        mean_dice,
        iou_scores,
        dice_scores
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    dataloader,
    device,
    model_name
):

    print()
    print("=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)

    model.eval()

    total_correct = 0
    total_pixels = 0

    intersection = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    union = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    dice_intersection = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    dice_denominator = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Disable gradients
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(
            dataloader
        ):

            images = images.to(device)
            targets = targets.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            outputs = get_model_output(outputs)

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # ------------------------------------------------
            # Pixel accuracy
            # ------------------------------------------------

            total_correct += (
                predictions == targets
            ).sum().item()

            total_pixels += targets.numel()

            # ------------------------------------------------
            # Per-class metrics
            # ------------------------------------------------

            for class_id in range(NUM_CLASSES):

                pred_class = (
                    predictions == class_id
                )

                target_class = (
                    targets == class_id
                )

                true_positive = (
                    pred_class & target_class
                ).sum().item()

                pred_pixels = (
                    pred_class
                ).sum().item()

                target_pixels = (
                    target_class
                ).sum().item()

                union[class_id] += (
                    pred_pixels
                    + target_pixels
                    - true_positive
                )

                intersection[class_id] += (
                    true_positive
                )

                dice_intersection[class_id] += (
                    true_positive
                )

                dice_denominator[class_id] += (
                    pred_pixels
                    + target_pixels
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx + 1
            ) % 20 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(dataloader)} batches"
                )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    pixel_accuracy = (
        total_correct / total_pixels
    )

    iou_scores = []

    dice_scores = []

    for class_id in range(NUM_CLASSES):

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        if union[class_id] == 0:

            iou = 1.0

        else:

            iou = (
                intersection[class_id]
                / union[class_id]
            ).item()

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        if dice_denominator[class_id] == 0:

            dice = 1.0

        else:

            dice = (
                2.0
                * dice_intersection[class_id]
                / dice_denominator[class_id]
            ).item()

        iou_scores.append(iou)
        dice_scores.append(dice)

    mean_iou = sum(iou_scores) / NUM_CLASSES

    mean_dice = sum(dice_scores) / NUM_CLASSES

    elapsed_time = time.time() - start_time

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(f"Pixel Accuracy : {pixel_accuracy:.4f}")
    print(f"Mean IoU       : {mean_iou:.4f}")
    print(f"Mean Dice      : {mean_dice:.4f}")
    print(f"Evaluation Time: {elapsed_time:.2f} seconds")

    print()
    print("--- Per-Class IoU / Dice ---")

    class_names = [
        "background",
        "human_diver",
        "aquatic_plants",
        "wrecks_ruins",
        "robots",
        "reefs_invertebrates",
        "fish_vertebrates",
        "sea_floor_rocks"
    ]

    for class_id in range(NUM_CLASSES):

        print(
            f"{class_names[class_id]:24s}"
            f"IoU: {iou_scores[class_id]:.4f}  "
            f"Dice: {dice_scores[class_id]:.4f}"
        )

    print("=" * 60)

    return {
        "model": model_name,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "iou_scores": iou_scores,
        "dice_scores": dice_scores
    }


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(model_name):

    if model_name == "U-Net":

        model = create_unet(
            num_classes=NUM_CLASSES
        )

    elif model_name == "DeepLabV3+":

        model = create_deeplabv3plus(
            num_classes=NUM_CLASSES
        )

    elif model_name == "SegFormer":

        model = create_segformer(
            num_classes=NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    class_names = [
        "background",
        "human_diver",
        "aquatic_plants",
        "wrecks_ruins",
        "robots",
        "reefs_invertebrates",
        "fish_vertebrates",
        "sea_floor_rocks"
    ]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header = [
        "model",
        "pixel_accuracy",
        "mean_iou",
        "mean_dice"
    ]

    for class_name in class_names:

        header.append(
            f"{class_name}_iou"
        )

        header.append(
            f"{class_name}_dice"
        )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        for result in results:

            row = [
                result["model"],
                f'{result["pixel_accuracy"]:.6f}',
                f'{result["mean_iou"]:.6f}',
                f'{result["mean_dice"]:.6f}'
            ]

            for class_id in range(NUM_CLASSES):

                row.append(
                    f'{result["iou_scores"][class_id]:.6f}'
                )

                row.append(
                    f'{result["dice_scores"][class_id]:.6f}'
                )

            writer.writerow(row)

    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(RESULTS_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SUIM MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print()
    print("Loading dataset...")

    dataset = SUIMDataset(
        DATASET_IMAGES,
        DATASET_MASKS,
        training=False
    )

    total_samples = len(dataset)

    # --------------------------------------------------------
    # Same 80/20 split
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
        [training_size, validation_size],
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
    # DataLoader
    # --------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = [
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

    all_results = []

    # ========================================================
    # EVALUATE EACH MODEL
    # ========================================================

    for model_name, checkpoint_name in models:

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            checkpoint_name
        )

        if not os.path.exists(
            checkpoint_path
        ):

            print()
            print(
                f"WARNING: Checkpoint not found:"
            )

            print(
                checkpoint_path
            )

            continue

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        print()
        print(
            f"Creating {model_name}..."
        )

        model = create_model(
            model_name
        )

        model = model.to(device)

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        model = load_checkpoint(
            model,
            checkpoint_path,
            device
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        result = evaluate_model(
            model,
            validation_loader,
            device,
            model_name
        )

        all_results.append(result)

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del model

        if device.type == "mps":

            torch.mps.empty_cache()

    # ========================================================
    # SAVE
    # ========================================================

    if len(all_results) == 0:

        raise RuntimeError(
            "No model checkpoints were found."
        )

    save_results(
        all_results
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"{'Model':20s}"
        f"{'Accuracy':>12s}"
        f"{'Mean IoU':>12s}"
        f"{'Mean Dice':>12s}"
    )

    print("-" * 60)

    for result in all_results:

        print(
            f"{result['model']:20s}"
            f"{result['pixel_accuracy']:12.4f}"
            f"{result['mean_iou']:12.4f}"
            f"{result['mean_dice']:12.4f}"
        )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = max(
        all_results,
        key=lambda x: x["mean_iou"]
    )

    print()
    print(
        f"BEST MODEL BY MEAN IoU: "
        f"{best_model['model']}"
    )

    print(
        f"Best Mean IoU: "
        f"{best_model['mean_iou']:.4f}"
    )

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()