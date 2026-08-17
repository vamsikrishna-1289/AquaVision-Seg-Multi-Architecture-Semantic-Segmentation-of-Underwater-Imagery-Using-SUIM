import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import time

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import SUIMDataset
from src.losses import get_loss_function
from src.models.segformer import create_segformer


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 8

IMAGE_DIR = (
    PROJECT_ROOT
    / "SUIM"
    / "train_val"
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "SUIM"
    / "train_val"
    / "masks"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "segformer_best.pth"
)

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

TRAIN_SPLIT = 0.8

RANDOM_SEED = 42


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
# BATCH METRICS
# ============================================================

def calculate_batch_metrics(
    predictions,
    targets
):
    """
    Calculate:

        Pixel Accuracy
        Mean IoU
        Mean Dice
    """

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    correct = (
        predicted_classes == targets
    ).sum().item()

    total = targets.numel()

    pixel_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    num_classes = predictions.shape[1]

    iou_scores = []
    dice_scores = []

    for class_id in range(num_classes):

        prediction_mask = (
            predicted_classes == class_id
        )

        target_mask = (
            targets == class_id
        )

        intersection = (
            prediction_mask
            & target_mask
        ).sum().item()

        prediction_area = (
            prediction_mask.sum().item()
        )

        target_area = (
            target_mask.sum().item()
        )

        union = (
            prediction_area
            + target_area
            - intersection
        )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        if union > 0:

            iou = (
                intersection / union
            )

            iou_scores.append(iou)

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        total_area = (
            prediction_area
            + target_area
        )

        if total_area > 0:

            dice = (
                2.0 * intersection
                / total_area
            )

            dice_scores.append(dice)

    mean_iou = (
        sum(iou_scores)
        / len(iou_scores)
        if iou_scores
        else 0.0
    )

    mean_dice = (
        sum(dice_scores)
        / len(dice_scores)
        if dice_scores
        else 0.0
    )

    return (
        pixel_accuracy,
        mean_iou,
        mean_dice
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    for images, masks in loader:

        images = images.to(device)

        masks = masks.to(device)

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predictions = model(images)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            predictions,
            masks
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    total_accuracy = 0.0
    total_iou = 0.0
    total_dice = 0.0

    num_batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)

            masks = masks.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            predictions = model(images)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                predictions,
                masks
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            (
                pixel_accuracy,
                mean_iou,
                mean_dice
            ) = calculate_batch_metrics(
                predictions,
                masks
            )

            total_accuracy += (
                pixel_accuracy
            )

            total_iou += (
                mean_iou
            )

            total_dice += (
                mean_dice
            )

            num_batches += 1

    validation_loss = (
        running_loss
        / len(loader.dataset)
    )

    pixel_accuracy = (
        total_accuracy
        / num_batches
    )

    mean_iou = (
        total_iou
        / num_batches
    )

    mean_dice = (
        total_dice
        / num_batches
    )

    return (
        validation_loss,
        pixel_accuracy,
        mean_iou,
        mean_dice
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SEGFORMER TRAINING")
    print("=" * 60)

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print(
        f"Device       : {device}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print("\nLoading dataset...")

    dataset = SUIMDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        training=True
    )

    total_samples = len(dataset)

    train_size = int(
        TRAIN_SPLIT
        * total_samples
    )

    validation_size = (
        total_samples
        - train_size
    )

    # --------------------------------------------------------
    # Reproducible split
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [
            train_size,
            validation_size
        ],
        generator=generator
    )

    print(
        f"Total samples     : {total_samples}"
    )

    print(
        f"Training samples  : {train_size}"
    )

    print(
        f"Validation samples: {validation_size}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_segformer(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = get_loss_function(
        num_classes=NUM_CLASSES,
        ce_weight=1.0,
        dice_weight=1.0
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_iou = 0.0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            validation_loss,
            pixel_accuracy,
            mean_iou,
            mean_dice
        ) = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            mean_iou
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss  : "
            f"{validation_loss:.4f}"
        )

        print(
            f"Pixel Accuracy   : "
            f"{pixel_accuracy:.4f}"
        )

        print(
            f"Mean IoU         : "
            f"{mean_iou:.4f}"
        )

        print(
            f"Mean Dice        : "
            f"{mean_dice:.4f}"
        )

        print(
            f"Learning Rate     : "
            f"{current_lr:.6f}"
        )

        print(
            f"Time              : "
            f"{elapsed_time:.2f} seconds"
        )

        print(
            "-" * 60
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if mean_iou > best_iou:

            best_iou = mean_iou

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "num_classes": NUM_CLASSES,
                },
                BEST_MODEL_PATH
            )

            print(
                f"✓ Best model saved "
                f"(IoU: {best_iou:.4f})"
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "SEGFORMER TRAINING COMPLETE"
    )

    print(
        f"Best Validation IoU: "
        f"{best_iou:.4f}"
    )

    print(
        "Best model saved to:"
    )

    print(
        BEST_MODEL_PATH
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()