"""Configuration classes for drawing utilities."""

from dataclasses import dataclass


@dataclass
class DrawConfig:
    """Base configuration for drawing."""
    pass


@dataclass
class DetectionConfig:
    """Configuration for drawing detection results."""
    pass


@dataclass
class PoseConfig:
    """Configuration for drawing pose estimation results."""
    pass
