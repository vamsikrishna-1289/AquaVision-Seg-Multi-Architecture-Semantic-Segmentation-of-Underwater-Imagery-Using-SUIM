from PIL import Image
from torchvision import transforms


IMAGE_SIZE = (256, 256)


def get_image_transform(training=False):
    """
    Create the image preprocessing pipeline.
    """

    transform_list = [
        transforms.Resize(
            IMAGE_SIZE,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    return transforms.Compose(transform_list)


def preprocess_image(image, training=False):
    """
    Preprocess an underwater RGB image.

    Output:
        Tensor of shape [3, 256, 256]
    """

    transform = get_image_transform(training)

    return transform(image)


def resize_mask(mask, size=IMAGE_SIZE):
    """
    Resize a segmentation mask.

    Nearest-neighbor interpolation is used because
    segmentation labels are discrete class IDs.
    """

    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL Image")

    return mask.resize(
        size,
        resample=Image.Resampling.NEAREST
    )