import logging
from pathlib import Path
from typing import override

import numpy as np
import onnxruntime as ort

from vk_photostylist.sam.inference.base import BaseInferenceEngine

_LOGGER = logging.getLogger(__name__)

try:
    ort.preload_dlls(directory="")
except Exception as e:  # noqa: BLE001
    _LOGGER.warning(f"Unable to load CUDA DLL: {e}")


class ONNXInferenceEngine(BaseInferenceEngine):
    _providers = (
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    )

    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
    ):
        super().__init__(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
        )

        print(f"PROVIDERS: {ort.get_available_providers()}")

        self._sam_encoder = self._create_session(
            self._encoder_path,
        )
        self._sam_encoder_input_name = self._sam_encoder.get_inputs()[0].name
        self._sam_encoder_output_name = self._sam_encoder.get_outputs()[0].name

        self._sam_decoder = self._create_session(
            self._decoder_path,
        )
        self._sam_decoder_input_names = [
            input.name for input in self._sam_decoder.get_inputs()
        ]
        self._sam_decoder_output_names = [
            output.name for output in self._sam_decoder.get_outputs()
        ]

        self._image_embedding = None
        self._low_res_mask = None

    def _create_session(
        self,
        model_path: Path | str,
    ) -> ort.InferenceSession:
        session = ort.InferenceSession(
            model_path,
            providers=self._providers,
        )

        active_providers = session.get_providers()
        _LOGGER.info(f"Session created. Active providers: {active_providers}")

        return session

    @override
    def _extract_image_embedding(
        self,
        image: np.ndarray,
    ):
        return self._sam_encoder.run(
            [self._sam_encoder_output_name], {self._sam_encoder_input_name: image}
        )[0]

    @override
    def _extract_decoder_embedding(
        self,
        image_emb: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: np.ndarray,
        has_mask_input: np.ndarray,
        orig_imgsz: np.ndarray,
    ):
        masks, iou_predictions, low_res_masks = self._sam_decoder.run(
            self._sam_decoder_output_names,
            dict(
                zip(
                    self._sam_decoder_input_names,
                    [
                        image_emb,
                        point_coords,
                        point_labels,
                        mask_input,
                        has_mask_input,
                        orig_imgsz,
                    ],
                )
            ),
        )
        return masks, iou_predictions, low_res_masks

    @override
    def __call__(
        self,
        image: np.ndarray,
        point_coords: list[tuple[int, int]],
        point_labels: list[int],
    ):
        image, orig_size, new_size, scale = self._preprocess_image(image)
        self._image_embedding = self._extract_image_embedding(image)

        point_coords = [[c[0] * scale, c[1] * scale] for c in point_coords]
        point_coords.append([0.0, 0.0])
        point_labels.append(-1)

        point_coords = np.array([point_coords], dtype=np.float32)
        point_labels = np.array([point_labels], dtype=np.float32)

        outputs = self._extract_decoder_embedding(
            image_emb=self._image_embedding,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=np.zeros((1, 1, 256, 256), dtype=np.float32),
            has_mask_input=np.zeros((1), dtype=np.float32),
            orig_imgsz=np.array(orig_size, dtype=np.float32),
        )
        self._low_res_mask = outputs[-1]

        return self._postprocess_result(*outputs[:-1], orig_size, new_size)
