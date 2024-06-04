"""Module to turn smiles into images."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from PIL import Image


@dataclass
class BlocksToImage(TrainingBlock):
    """Combine the building blocks into a single image."""

    img_width: int = 100
    img_heigth: int = 50

    def train(
        self,
        X: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Turn the building block smiles into images.

        :param X: Array containing the building block smiles
        :param y: The labels for the molecules
        :return: Image
        """
        images = [self.concatenate_images(smiles, self.img_width, self.img_heigth) for smiles in X]
        return np.array(images).transpose(0, 3, 1, 2), y

    @staticmethod
    def concatenate_images(images: npt.NDArray[np.str_], img_width: int, img_heigth: int) -> npt.NDArray[np.float32]:
        """Concatenate the images into a single image.

        :param images: numpy arrays representing the images.
        """
        # Initialize the new image with the correct size
        concatenated = Image.new("RGB", (3 * img_width, img_heigth))

        # Initialize the offset of the first image
        x_offset = 0
        for image in images:
            img = Image.fromarray(image)
            concatenated.paste(img, (x_offset, 0))
            x_offset += img.width

        return np.array(concatenated)

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
