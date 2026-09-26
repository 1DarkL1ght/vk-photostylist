from typing import override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule


class ColorConversion(BaseModule):
    @override
    def __call__(
        self,
        image: np.ndarray | cv2.UMat,
        src_colorscheme: str,
        dst_colorscheme: str,
        prefix: str = "",
    ):
        prefix_str = "_" * int(len(prefix) > 0) + prefix
        return cv2.cvtColor(
            image,
            getattr(cv2, f"COLOR{prefix_str}_{src_colorscheme}2{dst_colorscheme}"),
        )
