"""Utility functions for drawing complete results."""

import numpy as np

from simple_rtmw.draw.config import DetectionConfig, DrawConfig, PoseConfig
from simple_rtmw.draw.detection import draw_detection_boxes
from simple_rtmw.draw.pose import draw_pose
from simple_rtmw.types import PoseResult


def draw_annotated_image(
    image: np.ndarray,
    detection_boxes: np.ndarray | None = None,
    pose_results: list[PoseResult] | None = None,
    draw_config: DrawConfig | None = None,
    detection_config: DetectionConfig | None = None,
    pose_config: PoseConfig | None = None,
) -> np.ndarray:
    """Draw complete pose estimation results on image.

    This function provides a high-level interface for drawing both detection boxes
    and pose annotations on an image. It orchestrates the drawing process by calling
    the appropriate specialized drawing functions based on the provided configuration.

    Args:
        image: Input image in BGR format (H, W, 3).
        detection_boxes: Detection bounding boxes in format (N, 4) where each box
            is [x1, y1, x2, y2]. Optional.
        pose_results: List of pose estimation results for detected persons. Optional.
        draw_config: Overall drawing configuration controlling what elements to draw.
            If None, uses default configuration.
        detection_config: Configuration for detection box appearance. If None, uses
            default configuration.
        pose_config: Configuration for pose keypoint and skeleton appearance. If None,
            uses default configuration.

    Returns:
        Annotated image with requested elements drawn.

    Raises:
        ValueError: If image format is invalid.

    Examples:
        >>> # Draw only detection boxes
        >>> annotated = draw_annotated_image(image, detection_boxes=boxes)

        >>> # Draw only pose results
        >>> annotated = draw_annotated_image(image, pose_results=poses)

        >>> # Draw both with custom configuration
        >>> config = DrawConfig(draw_detection_boxes=True, draw_pose_keypoints=True)
        >>> annotated = draw_annotated_image(image, boxes, poses, config)

        >>> # Disable skeleton drawing
        >>> config = DrawConfig(draw_pose_skeleton=False)
        >>> annotated = draw_annotated_image(image, pose_results=poses, draw_config=config)
    """
    # Validation
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be in BGR format (H, W, 3)")

    if draw_config is None:
        draw_config = DrawConfig()
    if detection_config is None:
        detection_config = DetectionConfig()
    if pose_config is None:
        pose_config = PoseConfig()

    result_image = image.copy() if draw_config.copy else image

    # Detection boxes
    if (draw_config.draw_detection_boxes and
        detection_boxes is not None and
        len(detection_boxes) > 0):
        result_image = draw_detection_boxes(
            result_image,
            detection_boxes,
            config=detection_config,
            copy=False,
        )

    # Pose Annotations
    if pose_results is not None and len(pose_results) > 0:
        result_image = draw_pose(
            result_image,
            pose_results,
            config=pose_config,
            copy=False,
            draw_keypoints=draw_config.draw_pose_keypoints,
            draw_skeleton=draw_config.draw_pose_skeleton,
        )

    return result_image
