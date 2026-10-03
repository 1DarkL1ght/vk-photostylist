from typing import override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule


class GaussianBlur(BaseModule):
    def __init__(self, kernel_size: int):
        super().__init__()

        self._kernel_size = (kernel_size, kernel_size)

    @override
    def __call__(
        self,
        image: np.ndarray | cv2.UMat,
        std_x: int = 0,
        std_y: int | None = None,
    ):
        image = super().__call__(image)

        return cv2.GaussianBlur(
            image,
            self._kernel_size,
            sigmaX=std_x,
            sigmaY=std_y,
        )
