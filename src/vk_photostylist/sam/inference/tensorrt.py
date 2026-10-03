import logging
from pathlib import Path

import onnxruntime as ort

from vk_photostylist.sam.inference.onnx import ONNXInferenceEngine

_LOGGER = logging.getLogger(__name__)


class TensorRTInferenceEngine(ONNXInferenceEngine):
    def _create_session(
        self,
        model_path: Path | str,
    ) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        CACHE_PATH = Path(".trt_cachce/")
        CACHE_PATH.mkdir(exist_ok=True)

        trt_provider_options = {
            "trt_fp16_enable": "True",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(CACHE_PATH.resolve()),
        }

        try:
            import os
            import sys

            venv_base = sys.prefix
            trt_libs_dir = os.path.join(
                venv_base, "Lib", "site-packages", "tensorrt_libs"
            )

            if os.path.exists(trt_libs_dir):
                os.add_dll_directory(trt_libs_dir)

            session = ort.InferenceSession(
                model_path,
                providers=[
                    ("TensorrtExecutionProvider", trt_provider_options),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                sess_options=options,
            )

            active_providers = session.get_providers()
            _LOGGER.info(f"Session created. Active providers: {active_providers}")

            return session
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning(f"TensorRT is not available, falling back to ONNX: {e}")
            return super()._create_session(model_path)
