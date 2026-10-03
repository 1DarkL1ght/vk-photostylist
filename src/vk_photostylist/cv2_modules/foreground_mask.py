from pathlib import Path
from typing import Literal, override

import cv2
import numpy as np

from vk_photostylist.cv2_modules.base import BaseModule
from vk_photostylist.sam.inference_engine import SAMInference


def _to_ndarray(src: np.ndarray | cv2.UMat):
    return src.get() if isinstance(src, cv2.UMat) else src


class ForegroundMask(BaseModule):
    NUM_POINTS = 5
    MIN_COMPONENT_AREA = 0.01
    CONTOUR_SAMPLE_SIZE = 200
    PERCENTILE_THRESH = 75

    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        backend: Literal["ONNX", "TensorRT"],
    ):
        super().__init__()

        self._sam = SAMInference(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            backend=backend,
        )

    def _find_mask_naive(self, image: np.ndarray | cv2.UMat):
        h, w = _to_ndarray(image).shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        edgeImg = cv2.Canny(blurred, 50, 150)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edgeImg = cv2.morphologyEx(edgeImg, cv2.MORPH_CLOSE, kernel_edge)

        _, sat_hsv, _ = cv2.split(hsv)
        _, l_lab, _ = cv2.split(lab)
        sat_hsv = _to_ndarray(sat_hsv)
        l_lab = _to_ndarray(l_lab)
        sat_score = np.clip((sat_hsv.astype(np.float32) / 255) ** 2, 0, 1)

        chroma_mask = ((sat_hsv > 50) & (l_lab > 60) & (l_lab < 220)).astype(np.float32)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        center_dist = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (h // 2)
        fg_proba = np.clip(1.0 - center_dist * 0.3, 0.2, 1.0)

        score = (
            0.3 * (_to_ndarray(edgeImg).astype(np.float32) / 255.0)
            + 0.4 * chroma_mask
            + 0.3 * fg_proba
        )
        score *= sat_score[:, :] + 0.05

        thresh_val = float(np.clip(np.percentile(score[score > 0], 65), 0.1, 0.99))
        _, binary = cv2.threshold(
            (score * 255).astype(np.uint8),
            round(thresh_val * 255),
            255,
            cv2.THRESH_BINARY,
        )

        _, labels = cv2.connectedComponents(binary)
        label_areas = np.bincount(labels.ravel())[1:]
        if len(label_areas) == 0 or np.max(label_areas) == 0:
            return np.zeros_like(binary)

        best_label_idx = np.argmax(label_areas)
        mask = np.zeros_like(binary)
        mask[labels == best_label_idx + 1] = 255

        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill)

        _, labels_after = cv2.connectedComponents(mask)
        label_areas_after = np.bincount(labels_after.ravel())[1:]
        if len(label_areas_after) == 0 or np.max(label_areas_after) == 0:
            return np.zeros_like(binary)
        best_after = np.argmax(label_areas_after)
        mask = (labels_after == best_after + 1).astype(np.uint8) * 255

        return mask

    def _get_bbox_ltrb(
        self, mask: np.ndarray
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        pts = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(pts)
        return (x, y), (x + w, y + h)

    @override
    def __call__(self, image: np.ndarray | cv2.UMat):
        image = super().__call__(image)

        naive_mask = self._find_mask_naive(image)

        sampled_points = self._get_bbox_ltrb(naive_mask)

        sam_mask = self._sam(
            image.get() if isinstance(image, cv2.UMat) else image,
            point_coords=sampled_points,
            point_labels=[2, 3],
        )

        return sam_mask
