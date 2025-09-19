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
    pass
