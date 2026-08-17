import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation.

    Dice Loss is useful when some segmentation classes
    occupy much less area than others.
    """

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Parameters
        ----------
        predictions : torch.Tensor
            Model output.

            Shape:
                [B, C, H, W]

        targets : torch.Tensor
            Ground-truth class IDs.

            Shape:
                [B, H, W]

        Returns
        -------
        torch.Tensor
            Dice loss.
        """

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = F.softmax(
            predictions,
            dim=1
        )

        # ----------------------------------------------------
        # Number of classes
        # ----------------------------------------------------

        num_classes = predictions.shape[1]

        # ----------------------------------------------------
        # Convert targets to one-hot encoding
        # ----------------------------------------------------

        targets_one_hot = F.one_hot(
            targets,
            num_classes=num_classes
        )

        # [B, H, W, C] → [B, C, H, W]

        targets_one_hot = targets_one_hot.permute(
            0,
            3,
            1,
            2
        ).float()

        # ----------------------------------------------------
        # Calculate intersection
        # ----------------------------------------------------

        intersection = (
            probabilities * targets_one_hot
        ).sum(
            dim=(0, 2, 3)
        )

        # ----------------------------------------------------
        # Calculate prediction area
        # ----------------------------------------------------

        prediction_area = probabilities.sum(
            dim=(0, 2, 3)
        )

        # ----------------------------------------------------
        # Calculate target area
        # ----------------------------------------------------

        target_area = targets_one_hot.sum(
            dim=(0, 2, 3)
        )

        # ----------------------------------------------------
        # Dice coefficient
        # ----------------------------------------------------

        dice_score = (
            2.0 * intersection
            + self.smooth
        ) / (
            prediction_area
            + target_area
            + self.smooth
        )

        # ----------------------------------------------------
        # Convert Dice score to loss
        # ----------------------------------------------------

        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


# ============================================================
# Combined Cross-Entropy + Dice Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss.

    Loss = CE + Dice

    Cross-Entropy:
        Encourages correct pixel classification.

    Dice:
        Encourages good overlap between predicted
        segmentation regions and ground truth.
    """

    def __init__(
        self,
        num_classes=8,
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=None,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.num_classes = num_classes

        self.ce_weight = ce_weight

        self.dice_weight = dice_weight

        # ----------------------------------------------------
        # Cross-Entropy Loss
        # ----------------------------------------------------

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights
        )

        # ----------------------------------------------------
        # Dice Loss
        # ----------------------------------------------------

        self.dice = DiceLoss()

    def forward(
        self,
        predictions,
        targets
    ):
        """
        Parameters
        ----------
        predictions : torch.Tensor
            Shape:
                [B, C, H, W]

        targets : torch.Tensor
            Shape:
                [B, H, W]

        Returns
        -------
        torch.Tensor
            Combined segmentation loss.
        """

        # ----------------------------------------------------
        # Cross-Entropy
        # ----------------------------------------------------

        ce_loss = self.cross_entropy(
            predictions,
            targets
        )

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice_loss = self.dice(
            predictions,
            targets
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss


# ============================================================
# Factory Function
# ============================================================

def get_loss_function(
    num_classes=8,
    ce_weight=1.0,
    dice_weight=1.0,
    class_weights=None,
):
    """
    Create the loss function used for training.

    Parameters
    ----------
    num_classes : int
        Number of SUIM segmentation classes.

    ce_weight : float
        Weight of Cross-Entropy Loss.

    dice_weight : float
        Weight of Dice Loss.

    class_weights : torch.Tensor or None
        Optional class weights for Cross-Entropy.

    Default:

        Loss = CrossEntropy + Dice
    """

    return CombinedLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        class_weights=class_weights,
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOSS FUNCTION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Simulated model configuration
    # --------------------------------------------------------

    batch_size = 2

    num_classes = 8

    height = 256

    width = 256

    # --------------------------------------------------------
    # Simulated model output
    # --------------------------------------------------------

    predictions = torch.randn(
        batch_size,
        num_classes,
        height,
        width
    )

    # --------------------------------------------------------
    # Simulated ground-truth masks
    # --------------------------------------------------------

    targets = torch.randint(
        0,
        num_classes,
        (
            batch_size,
            height,
            width
        )
    )

    # --------------------------------------------------------
    # Create loss
    # --------------------------------------------------------

    loss_function = get_loss_function(
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = loss_function(
        predictions,
        targets
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Predictions shape : {predictions.shape}"
    )

    print(
        f"Targets shape     : {targets.shape}"
    )

    print(
        f"Loss              : {loss.item():.6f}"
    )

    print("=" * 60)
    print("LOSS FUNCTION TEST COMPLETE")
    print("=" * 60)