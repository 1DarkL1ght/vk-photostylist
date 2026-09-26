from abc import ABC
from abc import abstractmethod
from pathlib import Path

import numpy as np
import cv2


class BaseInferenceEngine(ABC):
    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
    ):
        self._encoder_path = encoder_path
        self._decoder_path = decoder_path


    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]

        scale = 1024.0 / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img_resized = cv2.resize(image, (new_w, new_h))

        input_tensor = np.zeros((1024, 1024, 3), dtype=np.float32)
        input_tensor[:new_h, :new_w, :] = img_resized

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        input_tensor = (input_tensor - mean) / std

        return input_tensor, (orig_h, orig_w), (new_h, new_w), scale

    def _postprocess_result(
        self,
        masks,
        iou_predictions,
        orig_size: tuple[int, int],
        new_size: tuple[int, int],
    ) -> np.ndarray:
        orig_h, orig_w = orig_size
        new_h, new_w = new_size

        best_mask_idx = np.argmax(iou_predictions[0])
        best_mask_logits = masks[0, best_mask_idx, :new_h, :new_w]
        mask_resized = cv2.resize(
            best_mask_logits, 
            (orig_w, orig_h), 
            interpolation=cv2.INTER_LINEAR
        )
        binary_mask = (mask_resized > 0.0).astype(np.uint8) * 255
        return binary_mask


    @abstractmethod
    def __call__(
        self,
        data: np.ndarray
    ):
        raise NotImplementedError("__call__ is not implemented for class BaseInferenceEngine.")
