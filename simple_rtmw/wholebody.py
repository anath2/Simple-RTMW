"""Whole-body pose estimation pipeline combining detection and pose estimation."""

import logging
from pathlib import Path

import numpy as np

from simple_rtmw.detection import Detector
from simple_rtmw.pose import PoseEstimator
from simple_rtmw.types import BodyResult, FaceResult, HandResult, Keypoint, PoseResult


logger = logging.getLogger(__name__)

CONFIG = {
    "detector": "http://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "detector_input_size": (640, 640),
    "pose_estimator": "http://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
    "pose_estimator_input_size": (288, 384),
    "backend": "onnxruntime",
    "device": "mps",
}


class Wholebody:
    """Pipeline for whole-body pose estimation using YOLOX detector and RTMPose.

    This class combines person detection and pose estimation to perform
    whole-body keypoint detection including body, face, and hand keypoints.
    """

    def __init__(self, device: str = "cpu", models_base_dir: Path = Path("./models")):
        """Initialize the whole-body pose estimation pipeline.

        Args:
            device: Device to run inference on ('cpu', 'cuda', or 'mps').
            models_base_dir: Base directory for storing downloaded models.
        """
        self.det_model = Detector(
            model_url=CONFIG["detector"],
            model_base_dir=models_base_dir / "detector",
            model_input_size=CONFIG["detector_input_size"],
            device=device,
        )
        self.pose_model = PoseEstimator(
            model_url=CONFIG["pose_estimator"],
            model_base_dir=models_base_dir / "pose_estimator",
            model_input_size=CONFIG["pose_estimator_input_size"],
            device=device,
        )

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run pose estimation on input image.

        Args:
            image: Input image as numpy array in BGR format.

        Returns:
            A tuple containing:
                - keypoints: Array of detected keypoints for all persons.
                - scores: Confidence scores for each keypoint.
        """
        bboxes = self.det_model(image)
        # Convert to list of bounding boxes
        bboxes = list(bboxes) if len(bboxes) > 0 else None
        keypoints, scores = self.pose_model(image, bboxes=bboxes)
        return keypoints, scores

    @staticmethod
    def format_result(keypoints_info: np.ndarray) -> list[PoseResult]:
        """Format raw keypoints into structured pose results.

        Converts raw keypoint arrays into structured PoseResult objects
        containing body, face, and hand keypoints with proper indexing.

        Args:
            keypoints_info: Raw keypoints array with shape (N, 134, 3) where
                N is the number of detected persons, 134 is the total number
                of keypoints, and 3 represents (x, y, score).

        Returns:
            List of PoseResult objects for each detected person.
        """
        score_threshold = 0.3

        def create_null_keypoint(idx: int) -> Keypoint:
            return Keypoint(np.nan, np.nan, 0.0, idx)

        def format_keypoint_part(part: np.ndarray) -> list[Keypoint]:
            return [
                Keypoint(x, y, score, i) if score >= score_threshold else create_null_keypoint(i)
                for i, (x, y, score) in enumerate(part)
            ]

        def total_score(keypoints: list[Keypoint]) -> float:
            return sum(keypoint.score for keypoint in keypoints)

        pose_results = []

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(instance[:18])
            left_hand_keypoints = format_keypoint_part(instance[92:113])
            right_hand_keypoints = format_keypoint_part(instance[113:134])
            face_keypoints = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while RTMPose only
            # provides 68 points. Padding the last 2 points with body eye keypoints
            # left eye (body keypoint 14)
            face_keypoints.append(body_keypoints[14])
            # right eye (body keypoint 15)
            face_keypoints.append(body_keypoints[15])

            body = BodyResult(body_keypoints, total_score(body_keypoints), len(body_keypoints))
            left_hand = HandResult(left_hand_keypoints)
            right_hand = HandResult(right_hand_keypoints)
            face = FaceResult(face_keypoints)

            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results
