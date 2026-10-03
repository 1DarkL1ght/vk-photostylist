from pathlib import Path

import cv2
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-save-images",
        action="store_true",
        default=False,
        help="Disable image saving for visual confirmation.",
    )


@pytest.fixture
def save_images(request) -> bool:
    return not request.config.getoption("--no-save-images")


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def models_root(data_path: Path) -> Path:
    return data_path / "models" / "sam"


@pytest.fixture
def images_path(data_path: Path) -> Path:
    return data_path / "images"


@pytest.fixture
def model_path(request, models_root: Path) -> tuple[Path, Path]:
    model_name = request.param

    model_mapping = {
        "mobilesam_int8": (
            "mobilesam_int8/mobile_sam.encoder.quant.onnx",
            "mobilesam_int8/sam_vit_h_4b8939.decoder.quant.onnx",
        )
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}")

    return (
        models_root / model_mapping[model_name][0],
        models_root / model_mapping[model_name][1],
    )


@pytest.fixture
def test_image(request, images_path: Path) -> np.ndarray:
    return cv2.imread(images_path / request.param, cv2.IMREAD_UNCHANGED)
