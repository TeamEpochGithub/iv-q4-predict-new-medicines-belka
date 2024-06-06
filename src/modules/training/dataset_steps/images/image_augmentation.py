"""Custom sequential class for image augmentation."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from torchvision import transforms


@dataclass
class ImageAugmentation(TrainingBlock):
    """Perform image augmentations on the molecule images."""

    degree: float = 90
    p_vertical: float = 0.5
    p_horizontal: float = 0.5

    def train(
        self,
        X: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Perform image augmentations on the molecule images.

        :param X: numpy array representing the image
        :param y: The labels for the molecules
        :return: Image
        """
        # Define the random image transformations
        horizontal = transforms.RandomHorizontalFlip(p=self.p_horizontal)
        vertical = transforms.RandomVerticalFlip(p=self.p_vertical)
        rotations = transforms.RandomRotation(degrees=self.degree)

        # Convert the arrays to torch tensors and perform the transformations
        augment = transforms.Compose([horizontal, vertical, rotations])
        images = [augment(torch.from_numpy(image).float()) for image in X]

        return np.array(images), y

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return True
