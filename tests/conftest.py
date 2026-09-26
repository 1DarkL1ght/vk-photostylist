from pathlib import Path

import pytest


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def models_path(data_path: Path) -> Path:
    return data_path.parent.parent / "models"


@pytest.fixture
def images_path(data_path: Path) -> Path:
    return data_path / "images"
