"""Dataclasses for representing pose results."""
from dataclasses import dataclass


@dataclass(slots=True)
class Keypoint:
    """Keypoint Coordinates and Score."""
    x: float
    y: float
    score: float = 1.0
    id: int = -1


@dataclass(slots=True)
class BodyResult:
    """17 Body keypoints."""
    keypoints: list[Keypoint]


@dataclass(slots=True)
class HandResult:
    """21 Hand keypoints each hand."""
    keypoints: list[Keypoint]


@dataclass(slots=True)
class FaceResult:
    """68 Face keypoints."""
    keypoints: list[Keypoint]


@dataclass(slots=True)
class FootResult:
    """3 Foot keypoints."""
    keypoints: list[Keypoint]


@dataclass(slots=True)
class PoseResult:
    """Pose result for a single person.

    Attributes:
        body: Body keypoints.
        left_hand: Left hand keypoints.
        right_hand: Right hand keypoints.
        face: Face keypoints.
        left_foot: Left foot keypoints.
        right_foot: Right foot keypoints.
    """
    body: BodyResult
    left_hand: HandResult
    right_hand: HandResult
    face: FaceResult
    left_foot: FootResult
    right_foot: FootResult

