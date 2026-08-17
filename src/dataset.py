from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing import preprocess_image, resize_mask


class SUIMDataset(Dataset):

    CLASS_NAMES = [
        "background",
        "human_diver",
        "aquatic_plants",
        "wrecks_ruins",
        "robots",
        "reefs_invertebrates",
        "fish_vertebrates",
        "sea_floor_rocks",
    ]

    NUM_CLASSES = len(CLASS_NAMES)

    COLOR_TO_CLASS = {
        (0, 0, 0): 0,
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (0, 255, 255): 3,
        (255, 0, 0): 4,
        (255, 0, 255): 5,
        (255, 255, 0): 6,
        (255, 255, 255): 7,
    }

    def __init__(
        self,
        image_dir,
        mask_dir,
        training=False,
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.training = training

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found:\n{self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found:\n{self.mask_dir}"
            )

        self.image_files = sorted(
            [
                file
                for file in self.image_dir.iterdir()
                if file.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }
            ]
        )

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in:\n{self.image_dir}"
            )

        self.samples = []

        for image_path in self.image_files:

            mask_path = self.mask_dir / (
                f"{image_path.stem}.bmp"
            )

            if not mask_path.exists():

                possible_masks = [
                    self.mask_dir / f"{image_path.stem}.png",
                    self.mask_dir / f"{image_path.stem}.jpg",
                    self.mask_dir / f"{image_path.stem}.jpeg",
                ]

                for candidate in possible_masks:

                    if candidate.exists():
                        mask_path = candidate
                        break

            if mask_path.exists():

                self.samples.append(
                    (image_path, mask_path)
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid image-mask pairs were found."
            )

        print(
            f"SUIMDataset initialized with "
            f"{len(self.samples)} image-mask pairs."
        )

    def __len__(self):
        return len(self.samples)

    @classmethod
    def mask_to_class_indices(cls, mask):

        mask_array = np.array(mask)

        class_mask = np.zeros(
            mask_array.shape[:2],
            dtype=np.int64,
        )

        for color, class_id in cls.COLOR_TO_CLASS.items():

            color_array = np.array(color)

            matches = np.all(
                mask_array == color_array,
                axis=-1,
            )

            class_mask[matches] = class_id

        return class_mask

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("RGB")

        image = preprocess_image(
            image,
            training=self.training,
        )

        mask = resize_mask(mask)

        mask = self.mask_to_class_indices(mask)

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask

    @classmethod
    def get_class_names(cls):
        return cls.CLASS_NAMES

    @classmethod
    def get_num_classes(cls):
        return cls.NUM_CLASSES