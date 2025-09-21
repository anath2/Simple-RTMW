from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from simple_rtmw.draw.detection import draw_detection_boxes
from simple_rtmw.draw.pose import draw_pose
from simple_rtmw.wholebody import Wholebody


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    url = "https://live.staticflickr.com/141/401685338_759da4a49a.jpg"
    image = requests.get(url).content
    img_array = np.asarray(bytearray(image), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@pytest.fixture(scope="module")
def wholebody() -> Wholebody:
    return Wholebody(device="mps")


@pytest.mark.gpu
def test_draw_detection_boxes(image: np.ndarray, wholebody: Wholebody) -> None:
    """Test detection box drawing with real detector output."""
    # Get detection boxes from the detector
    boxes = wholebody.det_model(image)

    # Draw detection boxes on the image
    annotated_image = draw_detection_boxes(image, boxes)

    # Verify the output
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == image.shape
    assert annotated_image.dtype == image.dtype

    # Save the annotated image to data folder
    output_path = Path("data/detection_boxes_output.jpg")
    success = cv2.imwrite(str(output_path), annotated_image)
    assert success, f"Failed to save image to {output_path}"

    # Verify the file was created
    assert output_path.exists(), f"Output file not found at {output_path}"

    # Verify we detected some boxes
    assert len(boxes) > 0, "No detection boxes found"


@pytest.mark.gpu
def test_draw_pose(image: np.ndarray, wholebody: Wholebody) -> None:
    """Test pose drawing with real pose estimation output."""
    # Get keypoints and scores from wholebody pipeline
    keypoints, scores = wholebody(image)

    # Combine keypoints and scores for format_result (expects shape N, 134, 3)
    keypoints_with_scores = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)

    # Format the results into structured pose objects
    pose_results = wholebody.format_result(keypoints_with_scores)

    # Draw pose on the image
    annotated_image = draw_pose(image, pose_results)

    # Verify the output
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == image.shape
    assert annotated_image.dtype == image.dtype

    # Save the annotated image to data folder
    output_path = Path("data/pose_output.jpg")
    success = cv2.imwrite(str(output_path), annotated_image)
    assert success, f"Failed to save image to {output_path}"

    # Verify the file was created
    assert output_path.exists(), f"Output file not found at {output_path}"

    # Verify we detected some poses
    assert len(pose_results) > 0, "No poses found"
