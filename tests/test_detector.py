from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from simple_rtmw.detection import Detector


@pytest.fixture(scope='module')
def image() -> np.ndarray:
    url = 'https://live.staticflickr.com/141/401685338_759da4a49a.jpg'
    image = requests.get(url).content
    img_array = np.asarray(bytearray(image), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@pytest.fixture(scope='module')
def model_url() -> str:
    return 'http://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'


@pytest.fixture(scope='module')
def model_base_dir() -> Path:
    return Path('./models/detector')


@pytest.mark.gpu
def test_detector_e2e(image: np.ndarray, model_url: str, model_base_dir: Path) -> None:
    """End-to-end tests for detector"""
    detector = Detector(
        model_url=model_url,
        model_base_dir=model_base_dir,
        model_input_size=(640, 640),
        device='mps',
    )
    boxes = detector(image)
    assert isinstance(boxes, np.ndarray)

