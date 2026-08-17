import sys
from pathlib import Path

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

import time
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import SUIMDataset
from src.models.unet import create_unet
from src.losses import CombinedLoss
from src.metrics import calculate_metrics


# ============================================================
# Configuration
# ============================================================

TRAIN_IMAGES = PROJECT_ROOT / "SUIM" / "train_val" / "images"
TRAIN_MASKS = PROJECT_ROOT / "SUIM" / "train_val" / "masks"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

NUM_CLASSES = 8
IMAGE_SIZE = 256

BATCH_SIZE = 4
NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

VAL_SPLIT = 0.2

NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("U-NET TRAINING")
    print("=" * 60)

    print(f"Device       : {DEVICE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = SUIMDataset(
        image_dir=TRAIN_IMAGES,
        mask_dir=TRAIN_MASKS,
        training=True
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    total_size = len(dataset)

    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Total samples     : {total_size}")
    print(f"Training samples  : {train_size}")
    print(f"Validation samples: {val_size}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_unet(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nModel parameters: {total_parameters:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_val_iou = 0.0

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            predictions = model(images)

            # Loss
            loss = criterion(
                predictions,
                masks
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            # Progress
            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss /= len(train_loader)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        total_intersection = torch.zeros(
            NUM_CLASSES,
            device=DEVICE
        )

        total_union = torch.zeros(
            NUM_CLASSES,
            device=DEVICE
        )

        total_pred = torch.zeros(
            NUM_CLASSES,
            device=DEVICE
        )

        total_target = torch.zeros(
            NUM_CLASSES,
            device=DEVICE
        )

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                predictions = model(images)

                loss = criterion(
                    predictions,
                    masks
                )

                val_loss += loss.item()

                predicted_classes = torch.argmax(
                    predictions,
                    dim=1
                )

                # --------------------------------------------
                # Calculate per-class statistics
                # --------------------------------------------

                for class_id in range(NUM_CLASSES):

                    pred_class = (
                        predicted_classes == class_id
                    )

                    target_class = (
                        masks == class_id
                    )

                    intersection = (
                        pred_class & target_class
                    ).sum()

                    union = (
                        pred_class | target_class
                    ).sum()

                    total_intersection[class_id] += intersection
                    total_union[class_id] += union
                    total_pred[class_id] += pred_class.sum()
                    total_target[class_id] += target_class.sum()

        val_loss /= len(val_loader)

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        iou = (
            total_intersection /
            (total_union + 1e-7)
        )

        mean_iou = iou.mean().item()

        # ----------------------------------------------------
        # Pixel accuracy
        # ----------------------------------------------------

        pixel_accuracy = (
            total_intersection.sum() /
            (total_target.sum() + 1e-7)
        ).item()

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice = (
            2 * total_intersection /
            (
                total_pred +
                total_target +
                1e-7
            )
        )

        mean_dice = dice.mean().item()

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(mean_iou)

        epoch_time = time.time() - start_time

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n" + "-" * 60)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss     : {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Pixel Accuracy  : {pixel_accuracy:.4f}"
        )

        print(
            f"Mean IoU        : {mean_iou:.4f}"
        )

        print(
            f"Mean Dice       : {mean_dice:.4f}"
        )

        print(
            f"Learning Rate   : "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        print(
            f"Time            : {epoch_time:.2f} seconds"
        )

        print("-" * 60)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if mean_iou > best_val_iou:

            best_val_iou = mean_iou

            checkpoint_path = (
                CHECKPOINT_DIR /
                "unet_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_iou": best_val_iou,
                    "mean_dice": mean_dice,
                    "pixel_accuracy": pixel_accuracy,
                },
                checkpoint_path
            )

            print(
                f"✓ Best model saved: "
                f"{checkpoint_path}"
            )

    # ========================================================
    # Complete
    # ========================================================

    print("\n" + "=" * 60)

    print("U-NET TRAINING COMPLETE")

    print(
        f"Best Validation IoU: "
        f"{best_val_iou:.4f}"
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()