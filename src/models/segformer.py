import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SegFormer Components
# ============================================================

class MLP(nn.Module):
    """
    Linear projection used in the SegFormer decoder.
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.proj = nn.Linear(
            input_dim,
            embed_dim
        )

    def forward(self, x):
        return self.proj(x)


# ============================================================
# Overlap Patch Embedding
# ============================================================

class OverlapPatchEmbedding(nn.Module):
    """
    Converts an image/feature map into overlapping patches.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size=7,
        stride=4,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = self.proj(x)

        batch_size, channels, height, width = x.shape

        # Make tensor contiguous before attention
        x = x.flatten(2).transpose(1, 2).contiguous()

        x = self.norm(x)

        x = x.transpose(1, 2).contiguous()

        x = x.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return x


# ============================================================
# Efficient Self Attention
# ============================================================

class EfficientSelfAttention(nn.Module):
    """
    Efficient self-attention block.

    Note:
        This component is retained for architecture completeness.
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        reduction_ratio=1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio

        self.norm = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):

        batch_size, channels, height, width = x.shape

        tokens = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        tokens = self.norm(tokens).contiguous()

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        attended = (
            attended
            .contiguous()
            .transpose(1, 2)
            .contiguous()
        )

        attended = attended.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return attended


# ============================================================
# Mix Feed Forward Network
# ============================================================

class MixFFN(nn.Module):
    """
    Mix-FFN using depthwise convolution.
    """

    def __init__(
        self,
        embed_dim,
        expansion=4,
    ):
        super().__init__()

        hidden_dim = embed_dim * expansion

        self.fc1 = nn.Conv2d(
            embed_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Conv2d(
            hidden_dim,
            embed_dim,
            kernel_size=1,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.dwconv(x)

        x = self.activation(x)

        x = self.fc2(x)

        return x


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    SegFormer transformer block.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MixFFN(
            embed_dim=embed_dim
        )

    def forward(self, x):

        batch_size, channels, height, width = x.shape

        # ----------------------------------------------------
        # Convert feature map to tokens
        # ----------------------------------------------------

        tokens = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized = self.norm1(tokens).contiguous()

        # ----------------------------------------------------
        # Self Attention
        # ----------------------------------------------------

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        attention_output = attention_output.contiguous()

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        tokens = (
            tokens + attention_output
        ).contiguous()

        # ----------------------------------------------------
        # Convert back to feature map
        # ----------------------------------------------------

        x = (
            tokens
            .transpose(1, 2)
            .contiguous()
        )

        x = x.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        # ----------------------------------------------------
        # Mix-FFN
        # ----------------------------------------------------

        normalized = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        normalized = self.norm2(
            normalized
        ).contiguous()

        normalized = (
            normalized
            .transpose(1, 2)
            .contiguous()
        )

        normalized = normalized.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        # ----------------------------------------------------
        # MLP residual connection
        # ----------------------------------------------------

        x = (
            x + self.mlp(normalized)
        ).contiguous()

        return x


# ============================================================
# SegFormer Encoder Stage
# ============================================================

class EncoderStage(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim,
        num_heads,
        depth,
        stride,
    ):
        super().__init__()

        self.patch_embedding = OverlapPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=7,
            stride=stride,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        for block in self.blocks:
            x = block(x)

        return x


# ============================================================
# SegFormer Encoder
# ============================================================

class SegFormerEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.stage1 = EncoderStage(
            in_channels=3,
            embed_dim=32,
            num_heads=1,
            depth=1,
            stride=4,
        )

        self.stage2 = EncoderStage(
            in_channels=32,
            embed_dim=64,
            num_heads=2,
            depth=1,
            stride=2,
        )

        self.stage3 = EncoderStage(
            in_channels=64,
            embed_dim=160,
            num_heads=5,
            depth=1,
            stride=2,
        )

        self.stage4 = EncoderStage(
            in_channels=160,
            embed_dim=256,
            num_heads=8,
            depth=1,
            stride=2,
        )

    def forward(self, x):

        features = []

        x = self.stage1(x)
        features.append(x)

        x = self.stage2(x)
        features.append(x)

        x = self.stage3(x)
        features.append(x)

        x = self.stage4(x)
        features.append(x)

        return features


# ============================================================
# SegFormer Decoder
# ============================================================

class SegFormerDecoder(nn.Module):

    def __init__(
        self,
        num_classes=8,
        decoder_dim=128,
    ):
        super().__init__()

        self.linear1 = MLP(
            32,
            decoder_dim
        )

        self.linear2 = MLP(
            64,
            decoder_dim
        )

        self.linear3 = MLP(
            160,
            decoder_dim
        )

        self.linear4 = MLP(
            256,
            decoder_dim
        )

        self.fusion = nn.Conv2d(
            decoder_dim * 4,
            decoder_dim,
            kernel_size=1,
        )

        self.batch_norm = nn.BatchNorm2d(
            decoder_dim
        )

        self.activation = nn.ReLU(
            inplace=True
        )

        self.classifier = nn.Conv2d(
            decoder_dim,
            num_classes,
            kernel_size=1,
        )

    def project_feature(
        self,
        feature,
        projection,
        target_size,
    ):

        batch_size, channels, height, width = feature.shape

        feature = (
            feature.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        feature = projection(feature)

        feature = (
            feature
            .transpose(1, 2)
            .contiguous()
        )

        feature = feature.reshape(
            batch_size,
            -1,
            height,
            width,
        )

        feature = F.interpolate(
            feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return feature

    def forward(self, features):

        feature1 = features[0]
        feature2 = features[1]
        feature3 = features[2]
        feature4 = features[3]

        target_size = feature1.shape[-2:]

        feature1 = self.project_feature(
            feature1,
            self.linear1,
            target_size,
        )

        feature2 = self.project_feature(
            feature2,
            self.linear2,
            target_size,
        )

        feature3 = self.project_feature(
            feature3,
            self.linear3,
            target_size,
        )

        feature4 = self.project_feature(
            feature4,
            self.linear4,
            target_size,
        )

        x = torch.cat(
            [
                feature1,
                feature2,
                feature3,
                feature4,
            ],
            dim=1,
        ).contiguous()

        x = self.fusion(x)

        x = self.batch_norm(x)

        x = self.activation(x)

        x = self.classifier(x)

        return x


# ============================================================
# Complete SegFormer Model
# ============================================================

class SegFormer(nn.Module):

    def __init__(
        self,
        num_classes=8,
    ):
        super().__init__()

        self.encoder = SegFormerEncoder()

        self.decoder = SegFormerDecoder(
            num_classes=num_classes,
            decoder_dim=128,
        )

        self.num_classes = num_classes

    def forward(self, x):

        input_size = x.shape[-2:]

        features = self.encoder(x)

        x = self.decoder(features)

        # ----------------------------------------------------
        # Restore original resolution
        # ----------------------------------------------------

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


# ============================================================
# Model Factory
# ============================================================

def create_segformer(
    num_classes=8,
):
    """
    Create SegFormer for SUIM.
    """

    return SegFormer(
        num_classes=num_classes
    )


# ============================================================
# Model Test
# ============================================================

def main():

    print("=" * 60)
    print("SEGFORMER MODEL TEST")
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

    model = create_segformer(
        num_classes=8
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    input_tensor = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    print(
        f"Input shape  : {input_tensor.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

    print(
        f"Output shape : {output.shape}"
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

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Output validation
    # --------------------------------------------------------

    expected_shape = (
        2,
        8,
        256,
        256,
    )

    if tuple(output.shape) == expected_shape:

        print(
            "✓ Output shape is correct."
        )

    else:

        print(
            "✗ Output shape is incorrect."
        )

        print(
            f"Expected: {expected_shape}"
        )

    print("=" * 60)
    print("SEGFORMER MODEL TEST COMPLETE")
    print("=" * 60)


# ============================================================
# Run test
# ============================================================

if __name__ == "__main__":
    main()