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
        model_type: Literal["base_fp16", "base-int8"],
        backend: Literal["ONNX", "TensorRT"],
        models_root: Path | str = "models",
    ):
        self._encoder_path = Path(models_root) / "sam"
        self._decoder_path = Path(models_root) / "sam"
        if model_type == "base_fp16":
            self._encoder_path /= Path("base_fp16") / "sam_vit_b_01ec64.encoder.onnx"
            self._decoder_path /= Path("base_fp16") / "sam_vit_b_01ec64.decoder.onnx"
        elif model_type == "base_int8":
            self._encoder_path /= Path("base_int8") / "sam_vit_b_01ec64.encoder.onnx"
            self._decoder_path /= Path("base_int8") / "sam_vit_b_01ec64.decoder.onnx"

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
