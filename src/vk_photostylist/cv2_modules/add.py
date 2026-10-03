from typing import override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule


class Add(BaseModule):
    @override
    def __call__(
        self,
        image1: np.ndarray | cv2.UMat,
        image2: np.ndarray | cv2.UMat,
        mask: np.ndarray | cv2.UMat | None = None,
    ):
        image1 = super().__call__(image1)
        image2 = super().__call__(image2)
        mask = super().__call__(mask)

        return cv2.add(image1, image2, mask=mask)
