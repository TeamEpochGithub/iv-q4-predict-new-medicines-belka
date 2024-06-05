"""Custom sequential class for resized crop and sharpness."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from torchvision import transforms


@dataclass
class ImageResizedCrop(TrainingBlock):
    """Perform image transformations to improve the quality."""

    img_size: int = 64
    sharpness: float = 2

    def train(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Perform image transformations to improve the quality.

        :param X: numpy array representing the image
        :param y: The labels for the molecules
        :return: Image
        """
        # Define the random image transformations
        resized_crop = transforms.RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.8, 1))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert the arrays to torch tensors and perform the transformations
        augment = transforms.Compose([resized_crop, normalize])
        images = [augment(torch.from_numpy(image).float()) for image in X]

        return np.array(images), y

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
