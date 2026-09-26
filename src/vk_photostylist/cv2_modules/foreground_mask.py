from pathlib import Path
from typing import Literal, override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule
from vk_photostylist.sam.inference_engine import SAMInference


class ForegroundMask(BaseModule):
    NUM_POINTS = 5

    def __init__(
        self,
        model_type: Literal["base_fp16", "base_int8"],
        backend: Literal["ONNX", "TensorRT"],
        models_root: Path | str = "models",
    ):
        super().__init__()

        self._sam = SAMInference(
            model_type=model_type,
            backend=backend,
            models_root=models_root,
        )

    def _find_mask_naive(self, image: np.ndarray | cv2.UMat):
        def detect_edges(channel: np.ndarray | cv2.UMat):
            sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
            sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
            sobel = np.hypot(sobelX, sobelY)
            sobel[sobel > 255] = 255
            return sobel

        def findSignificantContours(
            edgeImg: np.ndarray | cv2.UMat,
        ):
            contours, heirarchy = cv2.findContours(
                edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            level1 = []
            for i, data in enumerate(heirarchy[0]):
                if data[3] == -1:
                    data = np.insert(data, 0, [i])
                    level1.append(data)
            significant = []
            tooSmall = edgeImg.size * 10 / 100
            for tupl in level1:
                contour = contours[tupl[0]]
                area = cv2.contourArea(contour)
                if area > tooSmall:
                    significant.append([contour, area])

            significant.sort(key=lambda x: x[1])
            return [x[0] for x in significant]

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edgeImg = np.max(
            np.array(
                [
                    detect_edges(blurred[:, :, 0]),
                    detect_edges(blurred[:, :, 1]),
                    detect_edges(blurred[:, :, 2]),
                ],
            ),
            axis=0,
        )
        mean = np.mean(edgeImg)
        edgeImg[edgeImg <= mean] = 0

        edgeImg_8u = np.asarray(edgeImg, np.uint8)
        significant_contour = findSignificantContours(edgeImg_8u)

        mask = edgeImg.copy()
        mask[mask > 0] = 0
        cv2.fillPoly(mask, significant_contour, 255)

        return mask

    @override
    def __call__(self, image: np.ndarray | cv2.UMat):
        naive_mask = self._find_mask_naive(image)

        coords = np.argwhere(naive_mask == 255)
        indices = np.random.choice(len(coords), size=self.NUM_POINTS, replace=False)

        sampled_coords = coords[indices]

        y_indices = sampled_coords[:, 0]
        x_indices = sampled_coords[:, 1]

        sampled_points = np.stack((x_indices, y_indices), axis=-1).tolist()

        sam_mask = self._sam(
            image.get() if isinstance(image, cv2.UMat) else image,
            point_coords=sampled_points,
            point_labels=[1] * self.NUM_POINTS,
        )
        return sam_mask
