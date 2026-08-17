import torch


# ============================================================
# Pixel Accuracy
# ============================================================

def pixel_accuracy(predictions, targets):
    """
    Calculate pixel accuracy.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions of shape [B, C, H, W].

    targets : torch.Tensor
        Ground-truth class IDs of shape [B, H, W].

    Returns
    -------
    float
        Pixel accuracy.
    """

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    correct = (
        predicted_classes == targets
    ).sum().item()

    total = targets.numel()

    if total == 0:
        return 0.0

    return correct / total


# ============================================================
# Confusion Matrix
# ============================================================

def confusion_matrix(predictions, targets, num_classes):
    """
    Calculate the confusion matrix for semantic segmentation.

    Rows    = ground-truth classes
    Columns = predicted classes

    Parameters
    ----------
    predictions : torch.Tensor
        Shape [B, C, H, W].

    targets : torch.Tensor
        Shape [B, H, W].

    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    torch.Tensor
        Confusion matrix of shape [num_classes, num_classes].
    """

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    predicted_classes = predicted_classes.reshape(-1)
    targets = targets.reshape(-1)

    # Remove invalid target labels
    valid = (
        (targets >= 0) &
        (targets < num_classes)
    )

    predicted_classes = predicted_classes[valid]
    targets = targets[valid]

    # Convert pairs into indices
    indices = (
        targets * num_classes
        + predicted_classes
    )

    matrix = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    )

    matrix = matrix.reshape(
        num_classes,
        num_classes
    )

    return matrix


# ============================================================
# IoU
# ============================================================

def intersection_over_union(
    predictions,
    targets,
    num_classes
):
    """
    Calculate IoU for every class.

    IoU = TP / (TP + FP + FN)

    Returns
    -------
    torch.Tensor
        IoU for each class.
    """

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes
    ).float()

    true_positive = torch.diag(matrix)

    false_positive = (
        matrix.sum(dim=0)
        - true_positive
    )

    false_negative = (
        matrix.sum(dim=1)
        - true_positive
    )

    denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = torch.zeros_like(
        true_positive
    )

    valid = denominator > 0

    iou[valid] = (
        true_positive[valid]
        / denominator[valid]
    )

    return iou


# ============================================================
# Mean IoU
# ============================================================

def mean_iou(
    predictions,
    targets,
    num_classes
):
    """
    Calculate mean Intersection over Union.

    Classes that do not appear in the ground truth
    or prediction are excluded from the mean.

    Returns
    -------
    float
        Mean IoU.
    """

    ious = intersection_over_union(
        predictions,
        targets,
        num_classes
    )

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes
    ).float()

    class_presence = (
        matrix.sum(dim=0)
        + matrix.sum(dim=1)
    ) > 0

    if class_presence.sum() == 0:
        return 0.0

    return ious[class_presence].mean().item()


# ============================================================
# Dice Score
# ============================================================

def dice_score(
    predictions,
    targets,
    num_classes
):
    """
    Calculate Dice/F1 score for every class.

    Dice = 2TP / (2TP + FP + FN)

    Returns
    -------
    torch.Tensor
        Dice score for each class.
    """

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes
    ).float()

    true_positive = torch.diag(matrix)

    false_positive = (
        matrix.sum(dim=0)
        - true_positive
    )

    false_negative = (
        matrix.sum(dim=1)
        - true_positive
    )

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    dice = torch.zeros_like(
        true_positive
    )

    valid = denominator > 0

    dice[valid] = (
        2 * true_positive[valid]
        / denominator[valid]
    )

    return dice


# ============================================================
# Mean Dice Score
# ============================================================

def mean_dice(
    predictions,
    targets,
    num_classes
):
    """
    Calculate mean Dice score.
    """

    dice = dice_score(
        predictions,
        targets,
        num_classes
    )

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes
    ).float()

    class_presence = (
        matrix.sum(dim=0)
        + matrix.sum(dim=1)
    ) > 0

    if class_presence.sum() == 0:
        return 0.0

    return dice[class_presence].mean().item()


# ============================================================
# Per-Class Metrics
# ============================================================

def get_per_class_metrics(
    predictions,
    targets,
    class_names
):
    """
    Calculate IoU and Dice score for every class.

    Returns
    -------
    dict
        Dictionary containing per-class metrics.
    """

    num_classes = len(class_names)

    ious = intersection_over_union(
        predictions,
        targets,
        num_classes
    )

    dice = dice_score(
        predictions,
        targets,
        num_classes
    )

    results = {}

    for index, class_name in enumerate(
        class_names
    ):
        results[class_name] = {
            "iou": ious[index].item(),
            "dice": dice[index].item(),
        }

    return results


# ============================================================
# Complete Metric Report
# ============================================================

def calculate_metrics(
    predictions,
    targets,
    class_names
):
    """
    Calculate all major segmentation metrics.

    Returns
    -------
    dict
        Complete metric report.
    """

    num_classes = len(class_names)

    pixel_acc = pixel_accuracy(
        predictions,
        targets
    )

    miou = mean_iou(
        predictions,
        targets,
        num_classes
    )

    mdice = mean_dice(
        predictions,
        targets,
        num_classes
    )

    per_class = get_per_class_metrics(
        predictions,
        targets,
        class_names
    )

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": miou,
        "mean_dice": mdice,
        "per_class": per_class,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SEGMENTATION METRICS TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Fake model output
    # --------------------------------------------------------

    batch_size = 2
    num_classes = 8
    height = 256
    width = 256

    predictions = torch.randn(
        batch_size,
        num_classes,
        height,
        width
    )

    targets = torch.randint(
        0,
        num_classes,
        (
            batch_size,
            height,
            width
        )
    )

    class_names = [
        "background",
        "human_diver",
        "aquatic_plants",
        "wrecks_ruins",
        "robots",
        "reefs_invertebrates",
        "fish_vertebrates",
        "sea_floor_rocks",
    ]

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    results = calculate_metrics(
        predictions,
        targets,
        class_names
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"Pixel Accuracy : "
        f"{results['pixel_accuracy']:.4f}"
    )

    print(
        f"Mean IoU       : "
        f"{results['mean_iou']:.4f}"
    )

    print(
        f"Mean Dice      : "
        f"{results['mean_dice']:.4f}"
    )

    print("\n--- Per-Class Metrics ---")

    for class_name, values in results[
        "per_class"
    ].items():

        print(
            f"{class_name:25s} "
            f"IoU: {values['iou']:.4f}  "
            f"Dice: {values['dice']:.4f}"
        )

    print("\n" + "=" * 60)
    print("SEGMENTATION METRICS TEST COMPLETE")
    print("=" * 60)