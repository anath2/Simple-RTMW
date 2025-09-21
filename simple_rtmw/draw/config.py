"""Configuration classes for drawing utilities."""

from dataclasses import dataclass


@dataclass
class DrawConfig:
    """Base configuration for drawing."""
    pass


@dataclass
class DetectionConfig:
    """Configuration for drawing detection results."""
    box_color: tuple[int, int, int] = (0, 255, 0)  # BGR green
    box_thickness: int = 2


@dataclass
class PoseConfig:
    """Configuration for drawing pose estimation results."""
    # Keypoint visualization
    keypoint_radius: int = 3
    keypoint_thickness: int = -1  # filled circles

    # Body part colors (BGR)
    body_color: tuple[int, int, int] = (0, 255, 0)      # Green
    face_color: tuple[int, int, int] = (255, 0, 0)      # Blue
    left_hand_color: tuple[int, int, int] = (0, 0, 255) # Red
    right_hand_color: tuple[int, int, int] = (255, 255, 0) # Cyan

    # Skeleton connections
    draw_skeleton: bool = True
    skeleton_thickness: int = 2
    skeleton_color: tuple[int, int, int] = (255, 255, 255) # White
