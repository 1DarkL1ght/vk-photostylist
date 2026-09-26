from pathlib import Path

import cv2
import numpy as np

from vk_photostylist.cv2_modules import BackgroundBlur
from vk_photostylist.cv2_modules import BackgroundRemove
from vk_photostylist.cv2_modules import ColorConversion
from vk_photostylist.cv2_modules import GaussianBlur
from vk_photostylist.cv2_modules.foreground_mask import ForegroundMask


def test_foreground_mask_finder(images_path: Path, models_path: Path):
    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)

    foreground_mask_finder = ForegroundMask(
        model_type="base_fp16",
        backend="ONNX",
        models_root=models_path,
    )

    mask = foreground_mask_finder(input_image)

    print("MASK", mask)

def test_background_blur(images_path: Path, models_path: Path):
    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)

    blurrer = BackgroundBlur(
        model_type="base_fp16",
        backend="ONNX",
        kernel_size=5,
        models_root=models_path,
    )

    result = blurrer(input_image)

    cv2.imwrite("background_blur_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 3


def test_background_remove(images_path: Path, models_path: Path):
    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)

    background_remover = BackgroundRemove(
        model_type="base_fp16",
        backend="ONNX",
        models_root=models_path,
    )

    result = background_remover(input_image)

    cv2.imwrite("background_remove_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 3
    assert all(result[50, 50] == 0)


def test_color_conversion(images_path: Path):
    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)

    color_converter = ColorConversion()

    result = color_converter(input_image, "RGB", "GRAY")

    cv2.imwrite("color_conversion_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 2


def test_gaussian_blur(images_path: Path):
    input_image = cv2.imread(images_path / "car.jpg", cv2.IMREAD_UNCHANGED)

    blurrer = GaussianBlur(kernel_size=5)

    result = blurrer(input_image)

    cv2.imwrite("gaussian_blur_test.jpg", result)

    result = result.get() if isinstance(result, cv2.UMat) else result

    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 3
