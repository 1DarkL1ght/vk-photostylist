from pathlib import Path
from typing import Literal, override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule
from vk_photostylist.cv2_modules.foreground_mask import ForegroundMask
from vk_photostylist.cv2_modules.gaussian_blur import GaussianBlur


class BackgroundBlur(BaseModule):
    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        backend: Literal["ONNX", "TensorRT"],
        kernel_size: int,
    ):
        super().__init__()

        self._foreground_mask_finder = ForegroundMask(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            backend=backend,
        )
        self._blurrer = GaussianBlur(kernel_size)

    @override
    def __call__(
        self,
        image: np.ndarray | cv2.UMat,
    ):
        image = super().__call__(image)

        foreground_mask = self._foreground_mask_finder(image)
        blurred_image = self._blurrer(image)

        fg = cv2.bitwise_and(image, image, mask=foreground_mask)
        background_mask = cv2.bitwise_not(foreground_mask)
        bg = cv2.bitwise_and(blurred_image, blurred_image, mask=background_mask)

        result = cv2.add(fg, bg)

        return result
