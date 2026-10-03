from pathlib import Path

import cv2
import numpy as np
import pytest

from vk_photostylist.cv2_modules import (
    BackgroundBlur,
    BackgroundRemove,
    ColorConversion,
    GaussianBlur,
)
from vk_photostylist.cv2_modules.foreground_mask import ForegroundMask


@pytest.mark.parametrize(
    "model_path",
    ["mobilesam_int8"],
    indirect=True,
)
def test_foreground_mask_finder(images_path: Path, model_path: tuple[Path, Path]):
    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)

    encoder_path, decoder_path = model_path

    foreground_mask_finder = ForegroundMask(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        backend="ONNX",
    )

    _ = foreground_mask_finder(input_image)


@pytest.mark.parametrize(
    [
        "model_path",
        "image_name",
    ],
    [
        ("mobilesam_int8", "car.jpg"),
        ("mobilesam_int8", "cat.jpg"),
    ],
    indirect=["model_path"],
)
def test_background_blur(
    images_path: Path,
    model_path: tuple[Path, Path],
    save_images: bool,
    image_name: str,
):
    input_image = cv2.imread(images_path / image_name, cv2.IMREAD_UNCHANGED)

    encoder_path, decoder_path = model_path

    blurrer = BackgroundBlur(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        backend="ONNX",
        kernel_size=5,
    )

    result = blurrer(input_image)

    if save_images:
        cv2.imwrite(f"background_blur_{Path(image_name).stem}_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 3


@pytest.mark.parametrize(
    [
        "model_path",
        "image_name",
    ],
    [
        ("mobilesam_int8", "car.jpg"),
        ("mobilesam_int8", "cat.jpg"),
    ],
    indirect=["model_path"],
)
def test_background_remove(
    images_path: Path,
    model_path: tuple[Path, Path],
    save_images: bool,
    image_name: str,
):
    input_image = cv2.imread(images_path / image_name, cv2.IMREAD_UNCHANGED)

    encoder_path, decoder_path = model_path

    background_remover = BackgroundRemove(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        backend="ONNX",
    )

    result = background_remover(input_image)

    if save_images:
        cv2.imwrite(f"background_remove_{image_name}_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 3
    assert all(result[50, 50] == 0)


@pytest.mark.parametrize(
    "image_name",
    [
        "car.jpg",
        "cat.jpg",
    ],
)
def test_color_conversion(images_path: Path, save_images: bool, image_name: str):
    input_image = cv2.imread(images_path / image_name, cv2.IMREAD_UNCHANGED)

    color_converter = ColorConversion()

    result = color_converter(input_image, "RGB", "GRAY")

    if save_images:
        cv2.imwrite(f"color_conversion_{image_name}_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 2


@pytest.mark.parametrize(
    "image_name",
    [
        "car.jpg",
        "cat.jpg",
    ],
)
def test_gaussian_blur(images_path: Path, save_images: bool, image_name: str):
    input_image = cv2.imread(images_path / image_name, cv2.IMREAD_UNCHANGED)

    blurrer = GaussianBlur(kernel_size=5)

    result = blurrer(input_image)

    if save_images:
        cv2.imwrite(f"gaussian_blur_{image_name}_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 3
