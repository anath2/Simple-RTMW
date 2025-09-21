"""Drawing functions for pose estimation results."""

import cv2
import numpy as np

from simple_rtmw.draw.config import PoseConfig
from simple_rtmw.types import Keypoint, PoseResult


# Body skeleton connections (18 keypoints)
BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (15, 17), (16, 17),  # Feet connections (if applicable)
]

# Hand skeleton connections (21 keypoints each)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _is_valid_keypoint(keypoint: Keypoint) -> bool:
    """Check if keypoint is valid (not NaN or negative coordinates)."""
    return not (np.isnan(keypoint.x) or np.isnan(keypoint.y) or keypoint.x < 0 or keypoint.y < 0)


def draw_pose_keypoints(
    image: np.ndarray,
    pose_results: list[PoseResult],
    config: PoseConfig | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw pose keypoints on image."""
    if config is None:
        config = PoseConfig()

    result_image = image.copy() if copy else image

    for pose_result in pose_results:
        # Draw body keypoints
        _draw_keypoints_part(result_image, pose_result.body.keypoints, config.body_color, config)

        # Draw face keypoints
        _draw_keypoints_part(result_image, pose_result.face.keypoints, config.face_color, config)

        # Draw hand keypoints
        _draw_keypoints_part(result_image, pose_result.left_hand.keypoints, config.left_hand_color, config)
        _draw_keypoints_part(result_image, pose_result.right_hand.keypoints, config.right_hand_color, config)

    return result_image


def draw_pose_skeleton(
    image: np.ndarray,
    pose_results: list[PoseResult],
    config: PoseConfig | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw pose skeleton connections on image."""
    if config is None:
        config = PoseConfig()

    result_image = image.copy() if copy else image

    for pose_result in pose_results:
        # Draw body skeleton
        _draw_skeleton_part(result_image, pose_result.body.keypoints, BODY_CONNECTIONS, config)

        # Draw hand skeletons
        _draw_skeleton_part(result_image, pose_result.left_hand.keypoints, HAND_CONNECTIONS, config)
        _draw_skeleton_part(result_image, pose_result.right_hand.keypoints, HAND_CONNECTIONS, config)

    return result_image


def _draw_keypoints_part(image: np.ndarray, keypoints: list, color: tuple[int, int, int], config: PoseConfig) -> None:
    """Draw keypoints for a specific body part."""
    for keypoint in keypoints:
        if not _is_valid_keypoint(keypoint):
            continue

        center = (int(keypoint.x), int(keypoint.y))
        cv2.circle(image, center, config.keypoint_radius, color, config.keypoint_thickness)


def _draw_skeleton_part(
    image: np.ndarray,
    keypoints: list,
    connections: list[tuple[int, int]],
    config: PoseConfig,
) -> None:
    """Draw skeleton connections for a specific body part."""
    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_kp = keypoints[start_idx]
        end_kp = keypoints[end_idx]

        # Skip if either keypoint is invalid
        if not _is_valid_keypoint(start_kp) or not _is_valid_keypoint(end_kp):
            continue

        start_point = (int(start_kp.x), int(start_kp.y))
        end_point = (int(end_kp.x), int(end_kp.y))
        cv2.line(image, start_point, end_point, config.skeleton_color, config.skeleton_thickness)


def draw_pose(
    image: np.ndarray,
    pose_results: list[PoseResult],
    config: PoseConfig | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw complete pose including keypoints and skeleton connections."""
    if config is None:
        config = PoseConfig()

    result_image = image.copy() if copy else image

    # Draw skeleton first (so keypoints appear on top)
    if config.draw_skeleton:
        result_image = draw_pose_skeleton(result_image, pose_results, config, copy=False)

    # Draw keypoints on top
    return draw_pose_keypoints(result_image, pose_results, config, copy=False)
