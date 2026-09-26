from pathlib import Path

import cv2
import numpy as np

from vk_photostylist.sam import SAMInference


def test_onnx_sam(images_path: Path, models_path: Path):
    sam = SAMInference(
        model_type="base_fp16",
        backend="ONNX",
        models_root=models_path,
    )

    input_image = cv2.imread(images_path / "cat.jpg", cv2.IMREAD_UNCHANGED)
    point_coords = [(370, 250), (200, 200)]
    point_labels = [1, 1]

    out = sam(
        image=input_image,
        point_coords=point_coords,
        point_labels=point_labels,
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == input_image.shape[:-1]
