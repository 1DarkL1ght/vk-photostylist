from pathlib import Path
from typing import Literal, override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule
from vk_photostylist.cv2_modules.foreground_mask import ForegroundMask


class BackgroundRemove(BaseModule):
    def __init__(
        self,
        model_type: Literal["base_fp16", "base_int8"],
        backend: Literal["ONNX", "TensorRT"],
        models_root: Path | str = "models",
    ):
        super().__init__()

        self._foreground_mask_finder = ForegroundMask(
            model_type=model_type,
            backend=backend,
            models_root=models_root,
        )

    @override
    def __call__(
        self,
        image: np.ndarray | cv2.UMat,
    ):
        foreground_mask = self._foreground_mask_finder(image)
        if cv2.ocl.useOpenCL():
            image = cv2.UMat(image)
            foreground_mask = cv2.UMat(foreground_mask)

        result = cv2.bitwise_and(image, image, mask=foreground_mask)

        return result
