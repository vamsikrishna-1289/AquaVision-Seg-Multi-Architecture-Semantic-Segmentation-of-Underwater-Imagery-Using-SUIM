import torch
import torch.nn as nn


# ============================================================
# Double Convolution Block
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = DoubleConv(3, 64)

        self.enc2 = DoubleConv(64, 128)

        self.enc3 = DoubleConv(128, 256)

        self.enc4 = DoubleConv(256, 512)

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(512, 1024)

        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # ----------------------------------------------------
        # Final segmentation layer
        # ----------------------------------------------------

        self.final = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # Segmentation output
        output = self.final(d1)

        return output


# ============================================================
# Factory Function
# ============================================================

def create_unet(num_classes=8):

    """
    Create a U-Net model for SUIM semantic segmentation.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    UNet
        Initialized U-Net model.
    """

    return UNet(
        num_classes=num_classes
    )


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("U-NET MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_unet(
        num_classes=8
    ).to(device)

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        3,
        256,
        256
    ).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

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
        f"Total parameters     : {total_parameters:,}"
    )

    print(
        f"Trainable parameters : {trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    expected_shape = (
        2,
        8,
        256,
        256
    )

    if tuple(output.shape) == expected_shape:

        print("✓ Output shape is correct.")

    else:

        print(
            "✗ Output shape is incorrect."
        )

    print("=" * 60)
    print("U-NET MODEL TEST COMPLETE")
    print("=" * 60)