from pathlib import Path
from typing import Literal

import numpy as np

from vk_photostylist.sam.inference import (
    BaseInferenceEngine,
    ONNXInferenceEngine,
    TensorRTInferenceEngine,
)


class SAMInference:
    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        backend: Literal["ONNX", "TensorRT"],
    ):
        self._encoder_path = Path(encoder_path)
        self._decoder_path = Path(decoder_path)

        self._engine: BaseInferenceEngine | None = None
        if backend == "ONNX":
            self._engine = ONNXInferenceEngine(self._encoder_path, self._decoder_path)
        elif backend == "TensorRT":
            self._engine = TensorRTInferenceEngine(
                self._encoder_path, self._decoder_path
            )
        else:
            raise ValueError(
                f"Got unsupported backend {backend}. Supported backends are: ONNX, TensorRT"
            )

    def __call__(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ):
        return self._engine(
            image,
            point_coords,
            point_labels,
        )
