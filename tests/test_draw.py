from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from simple_rtmw.draw import (
    DetectionConfig,
    DrawConfig,
    PoseConfig,
    draw_annotated_image,
    draw_detection_boxes,
)
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


@pytest.mark.gpu
def test_draw_annotated_image_complete(image: np.ndarray, wholebody: Wholebody) -> None:
    """Test end-to-end drawing with both detection boxes and pose estimation."""
    # Get detection boxes and pose results
    boxes = wholebody.det_model(image)
    keypoints, scores = wholebody(image)

    # Format pose results
    keypoints_with_scores = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
    pose_results = wholebody.format_result(keypoints_with_scores)

    # Draw complete annotation
    annotated_image = draw_annotated_image(
        image,
        detection_boxes=boxes,
        pose_results=pose_results,
    )

    # Verify output
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == image.shape
    assert annotated_image.dtype == image.dtype

    # Save the result
    output_path = Path("data/complete_annotation_output.jpg")
    success = cv2.imwrite(str(output_path), annotated_image)
    assert success, f"Failed to save image to {output_path}"
    assert output_path.exists(), f"Output file not found at {output_path}"

    # Verify we have both detections and poses
    assert len(boxes) > 0, "No detection boxes found"
    assert len(pose_results) > 0, "No poses found"


@pytest.mark.gpu
def test_draw_annotated_image_detection_only(image: np.ndarray, wholebody: Wholebody) -> None:
    """Test drawing with detection boxes only."""
    boxes = wholebody.det_model(image)

    # Configure to draw only detection boxes
    config = DrawConfig(
        draw_detection_boxes=True,
        draw_pose_keypoints=False,
        draw_pose_skeleton=False,
    )

    annotated_image = draw_annotated_image(
        image,
        detection_boxes=boxes,
        draw_config=config,
    )

    # Verify output
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == image.shape

    # Save the result
    output_path = Path("data/detection_only_output.jpg")
    success = cv2.imwrite(str(output_path), annotated_image)
    assert success and output_path.exists()


@pytest.mark.gpu
def test_draw_annotated_image_pose_only(image: np.ndarray, wholebody: Wholebody) -> None:
    """Test drawing with pose estimation only."""
    keypoints, scores = wholebody(image)
    keypoints_with_scores = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
    pose_results = wholebody.format_result(keypoints_with_scores)

    # Configure to draw only pose
    config = DrawConfig(
        draw_detection_boxes=False,
        draw_pose_keypoints=True,
        draw_pose_skeleton=True,
    )

    annotated_image = draw_annotated_image(
        image,
        pose_results=pose_results,
        draw_config=config,
    )

    # Verify output
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == image.shape

    # Save the result
    output_path = Path("data/pose_only_output.jpg")
    success = cv2.imwrite(str(output_path), annotated_image)
    assert success and output_path.exists()


@pytest.mark.gpu
def test_draw_annotated_image_configurations(image: np.ndarray, wholebody: Wholebody) -> None:
    """Test different configuration combinations."""
    # Get data
    boxes = wholebody.det_model(image)
    keypoints, scores = wholebody(image)
    keypoints_with_scores = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
    pose_results = wholebody.format_result(keypoints_with_scores)

    # Test 1: Keypoints only (no skeleton)
    config1 = DrawConfig(
        draw_detection_boxes=True,
        draw_pose_keypoints=True,
        draw_pose_skeleton=False,
    )
    annotated1 = draw_annotated_image(image, boxes, pose_results, config1)
    output_path1 = Path("data/keypoints_only_output.jpg")
    cv2.imwrite(str(output_path1), annotated1)

    # Test 2: Skeleton only (no keypoints)
    config2 = DrawConfig(
        draw_detection_boxes=True,
        draw_pose_keypoints=False,
        draw_pose_skeleton=True,
    )
    annotated2 = draw_annotated_image(image, boxes, pose_results, config2)
    output_path2 = Path("data/skeleton_only_output.jpg")
    cv2.imwrite(str(output_path2), annotated2)

    # Test 3: Custom colors
    detection_config = DetectionConfig(
        box_color=(0, 0, 255),  # Red boxes
        box_thickness=3,
    )
    pose_config = PoseConfig(
        body_color=(255, 255, 0),  # Cyan body
        keypoint_radius=5,
        skeleton_thickness=3,
    )
    annotated3 = draw_annotated_image(
        image, boxes, pose_results,
        detection_config=detection_config,
        pose_config=pose_config,
    )
    output_path3 = Path("data/custom_colors_output.jpg")
    cv2.imwrite(str(output_path3), annotated3)

    # Verify all outputs are valid
    for annotated in [annotated1, annotated2, annotated3]:
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == image.shape
        assert annotated.dtype == image.dtype


def test_draw_annotated_image_edge_cases() -> None:
    """Test edge cases and error handling."""
    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test 1: Empty inputs
    result = draw_annotated_image(test_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == test_image.shape

    # Test 2: Empty detection boxes
    empty_boxes = np.array([]).reshape(0, 4)
    result = draw_annotated_image(test_image, detection_boxes=empty_boxes)
    assert isinstance(result, np.ndarray)

    # Test 3: Empty pose results
    result = draw_annotated_image(test_image, pose_results=[])
    assert isinstance(result, np.ndarray)

    # Test 4: Invalid image format
    invalid_image = np.zeros((100, 100), dtype=np.uint8)  # Grayscale
    with pytest.raises(ValueError, match="Image must be in BGR format"):
        draw_annotated_image(invalid_image)

    # Test 5: In-place modification
    test_image_copy = test_image.copy()
    config = DrawConfig(copy=False)
    result = draw_annotated_image(test_image_copy, draw_config=config)
    # Result should be the same object when copy=False
    assert result is test_image_copy
