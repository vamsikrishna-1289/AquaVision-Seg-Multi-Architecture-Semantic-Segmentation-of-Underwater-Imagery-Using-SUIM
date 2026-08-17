import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


# ============================================================
# ASPP CONVOLUTION
# ============================================================

class ASPPConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# ASPP GLOBAL POOLING
# ============================================================

class ASPPPooling(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        size = x.shape[-2:]

        pooled = F.adaptive_avg_pool2d(
            x,
            output_size=1
        )

        pooled = self.conv(
            pooled
        )

        pooled = F.interpolate(
            pooled,
            size=size,
            mode="bilinear",
            align_corners=False
        )

        return pooled


# ============================================================
# ASPP
# ============================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Uses multiple dilation rates to capture
    information at different spatial scales.
    """

    def __init__(
        self,
        in_channels,
        out_channels=256,
        atrous_rates=(6, 12, 18)
    ):
        super().__init__()

        self.branches = nn.ModuleList()

        # ----------------------------------------------------
        # 1x1 convolution
        # ----------------------------------------------------

        self.branches.append(
            nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                ),

                nn.ReLU(
                    inplace=True
                )
            )
        )

        # ----------------------------------------------------
        # Dilated convolutions
        # ----------------------------------------------------

        for rate in atrous_rates:

            self.branches.append(
                ASPPConv(
                    in_channels,
                    out_channels,
                    rate
                )
            )

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.branches.append(
            ASPPPooling(
                in_channels,
                out_channels
            )
        )

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        self.project = nn.Sequential(

            nn.Conv2d(
                out_channels * len(self.branches),
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.1
            )
        )

    def forward(self, x):

        outputs = []

        for branch in self.branches:

            outputs.append(
                branch(x)
            )

        x = torch.cat(
            outputs,
            dim=1
        )

        return self.project(x)


# ============================================================
# DEEPLABV3+ DECODER
# ============================================================

class DeepLabV3PlusDecoder(nn.Module):

    def __init__(
        self,
        low_level_channels,
        num_classes
    ):
        super().__init__()

        # ----------------------------------------------------
        # Low-level feature projection
        # ----------------------------------------------------

        self.low_level_projection = nn.Sequential(

            nn.Conv2d(
                low_level_channels,
                48,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                48
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Conv2d(
                256 + 48,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1
            )
        )

    def forward(
        self,
        high_level_features,
        low_level_features
    ):

        # ----------------------------------------------------
        # Upsample high-level features
        # ----------------------------------------------------

        high_level_features = F.interpolate(
            high_level_features,
            size=low_level_features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Project low-level features
        # ----------------------------------------------------

        low_level_features = (
            self.low_level_projection(
                low_level_features
            )
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        features = torch.cat(
            [
                high_level_features,
                low_level_features
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        return self.decoder(
            features
        )


# ============================================================
# DEEPLABV3+
# ============================================================

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ semantic segmentation model.

    Encoder:
        ResNet-18

    ASPP:
        Multi-scale feature extraction

    Decoder:
        High-level + low-level feature fusion

    Output:
        [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes=8,
        pretrained=False
    ):
        super().__init__()

        # ----------------------------------------------------
        # ResNet-18 backbone
        # ----------------------------------------------------

        if pretrained:

            raise ValueError(
                "Pretrained weights are disabled "
                "for this project."
            )

        backbone = resnet18(
            weights=None
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.layer0 = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1

        self.layer2 = backbone.layer2

        self.layer3 = backbone.layer3

        self.layer4 = backbone.layer4

        # ----------------------------------------------------
        # ASPP
        # ----------------------------------------------------

        self.aspp = ASPP(
            in_channels=512,
            out_channels=256,
            atrous_rates=(6, 12, 18)
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=64,
            num_classes=num_classes
        )

    def forward(self, x):

        input_size = x.shape[-2:]

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x = self.layer0(x)

        low_level = self.layer1(x)

        x = self.layer2(
            low_level
        )

        x = self.layer3(
            x
        )

        x = self.layer4(
            x
        )

        # ----------------------------------------------------
        # ASPP
        # ----------------------------------------------------

        x = self.aspp(
            x
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        x = self.decoder(
            x,
            low_level
        )

        # ----------------------------------------------------
        # Restore original resolution
        # ----------------------------------------------------

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# MODEL FACTORY
# ============================================================

def create_deeplabv3plus(
    num_classes=8
):
    """
    Create a DeepLabV3+ model.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    DeepLabV3Plus
        Initialized model.
    """

    return DeepLabV3Plus(
        num_classes=num_classes,
        pretrained=False
    )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DEEPLABV3+ MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    if torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    elif torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_deeplabv3plus(
        num_classes=8
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        3,
        256,
        256,
        device=device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            x
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        f"Input shape  : {x.shape}"
    )

    print(
        f"Output shape : {output.shape}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    expected_shape = (
        2,
        8,
        256,
        256
    )

    if tuple(output.shape) == expected_shape:

        print(
            "✓ Output shape is correct."
        )

    else:

        print(
            "✗ Output shape is incorrect."
        )

    print("=" * 60)
    print(
        "DEEPLABV3+ MODEL TEST COMPLETE"
    )
    print("=" * 60)