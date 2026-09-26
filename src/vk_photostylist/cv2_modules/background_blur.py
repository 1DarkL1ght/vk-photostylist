from typing_extensions import override
from typing import Literal
from pathlib import Path

import numpy as np
import cv2

from vk_photostylist.cv2_modules.base import BaseModule
from vk_photostylist.cv2_modules.foreground_mask import ForegroundMask
from vk_photostylist.cv2_modules.gaussian_blur import GaussianBlur


class BackgroundBlur(BaseModule):
    def __init__(
        self,
        model_type: Literal["base_fp16", "base_int8"],
        backend: Literal["ONNX", "TensorRT"],
        kernel_size: int,
        models_root: Path | str = "models",
    ):
        super().__init__()

        self._foreground_mask_finder = ForegroundMask(
            model_type=model_type,
            backend=backend,
            models_root=models_root,
        )
        self._blurrer = GaussianBlur(kernel_size)


    @override
    def __call__(
        self,
        image: np.ndarray | cv2.UMat,
    ):
        foreground_mask = self._foreground_mask_finder(image)
        blurred_image = self._blurrer(image)

        if cv2.ocl.useOpenCL():
            blurred_image = cv2.UMat(blurred_image)
            image = cv2.UMat(image)
            foreground_mask = cv2.UMat(foreground_mask)

        fg = cv2.bitwise_and(image, image, mask=foreground_mask)
        background_mask = cv2.bitwise_not(foreground_mask)
        bg = cv2.bitwise_and(blurred_image, blurred_image, mask=background_mask)

        result = cv2.add(fg, bg)

        return result
