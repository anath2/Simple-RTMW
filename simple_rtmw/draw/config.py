"""Configuration classes for drawing utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DrawConfig:
    """Base configuration for drawing."""
    image: np.ndarray | None = None
    copy: bool = True
    draw_detection_boxes: bool = True
    draw_pose_keypoints: bool = True
    draw_pose_skeleton: bool = True


@dataclass(slots=True)
class DetectionConfig:
    """Configuration for drawing detection results."""
    box_color: tuple[int, int, int] = (0, 255, 0)  # BGR green
    box_thickness: int = 2


@dataclass(slots=True)
class PoseConfig:
    """Configuration for drawing pose estimation results."""
    # Keypoint visualization
    keypoint_radius: int = 3
    keypoint_thickness: int = -1  # filled circles
    min_score: float = 3.0

    # Body part colors (BGR)
    body_color: tuple[int, int, int] = (0, 255, 0)      # Green
    face_color: tuple[int, int, int] = (255, 0, 0)      # Blue
    left_hand_color: tuple[int, int, int] = (0, 0, 255) # Red
    right_hand_color: tuple[int, int, int] = (255, 255, 0) # Cyan
    left_foot_color: tuple[int, int, int] = (128, 0, 128) # Purple
    right_foot_color: tuple[int, int, int] = (0, 128, 128) # Dark Cyan

    # Skeleton connections
    skeleton_thickness: int = 2
    skeleton_color: tuple[int, int, int] = (255, 255, 255) # White
