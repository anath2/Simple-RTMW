import cv2
import numpy as np
import pytest
import requests

from simple_rtmw.wholebody import Wholebody


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    url = "https://live.staticflickr.com/141/401685338_759da4a49a.jpg"
    image = requests.get(url).content
    img_array = np.asarray(bytearray(image), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@pytest.mark.gpu
def test_wholebody_e2e(image: np.ndarray) -> None:
    whole_body = Wholebody(device="mps")
    keypoints, scores = whole_body(image)
    assert isinstance(keypoints, np.ndarray)
    assert isinstance(scores, np.ndarray)
