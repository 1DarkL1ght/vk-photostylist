from typing import override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule


class Inv(BaseModule):
    @override
    def __call__(
        self,
        src: np.ndarray | cv2.UMat,
        mask: np.ndarray | cv2.UMat | None = None,
    ):
        src = super().__call__(src)
        mask = super().__call__(mask)

        return cv2.bitwise_not(src, mask=mask)
