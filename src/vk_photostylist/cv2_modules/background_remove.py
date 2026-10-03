from pathlib import Path
from typing import Literal, override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule
from vk_photostylist.cv2_modules.foreground_mask import ForegroundMask


class BackgroundRemove(BaseModule):
    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        backend: Literal["ONNX", "TensorRT"],
    ):
        super().__init__()

        self._foreground_mask_finder = ForegroundMask(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            backend=backend,
        )

    @override
    def __call__(
        self,
        image: np.ndarray | cv2.UMat,
    ):
        image = super().__call__(image)

        foreground_mask = self._foreground_mask_finder(image)
        foreground_mask = cv2.UMat(foreground_mask)

        result = cv2.bitwise_and(image, image, mask=foreground_mask)

        return result
