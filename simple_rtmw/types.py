from dataclasses import dataclass


@dataclass(slots=True)
class Keypoint:
    x: float
    y: float
    score: float = 1.0
    id: int = -1


@dataclass(slots=True)
class BodyResult:
    keypoints: list[Keypoint]
    total_score: float = 0.0
    total_parts: int = 0


@dataclass(slots=True)
class HandResult:
    keypoints: list[Keypoint]


@dataclass(slots=True)
class FaceResult:
    keypoints: list[Keypoint]


@dataclass(slots=True)
class PoseResult:
    body: BodyResult
    left_hand: HandResult
    right_hand: HandResult
    face: FaceResult
