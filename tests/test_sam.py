from pathlib import Path

import cv2
import numpy as np
import pytest

from vk_photostylist.sam import SAMInference


@pytest.mark.parametrize(
    [
        "backend",
        "model_path",
    ],
    [
        ("ONNX", "mobilesam_int8"),
        ("TensorRT", "mobilesam_int8"),
    ],
    indirect=["model_path"],
)
def test_sam(
    images_path: Path,
    save_images: bool,
    model_path: tuple[Path, Path],
    backend: str,
):
    encoder_path, decoder_path = model_path

    sam = SAMInference(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        backend=backend,
    )

    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)
    point_coords = [(370, 250), (200, 200)]
    point_labels = [1, 1]

    out = sam(
        image=input_image,
        point_coords=point_coords,
        point_labels=point_labels,
    ).astype(bool)

    mask_color = np.array([255, 0, 255], dtype=np.uint8)
    alpha = 0.5

    input_image[out] = alpha * mask_color + (1 - alpha) * input_image[out]

    if save_images:
        cv2.imwrite(f"sam_{backend}_{encoder_path.parent.name}_test.jpg", input_image)

    assert isinstance(out, np.ndarray)
    assert out.shape == input_image.shape[:-1]
