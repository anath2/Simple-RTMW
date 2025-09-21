"""Whole-body pose estimation pipeline combining detection and pose estimation."""

import logging
from pathlib import Path

import numpy as np

from simple_rtmw.detection import Detector
from simple_rtmw.pose import PoseEstimator
from simple_rtmw.types import BodyResult, FaceResult, FootResult, HandResult, Keypoint, PoseResult


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
    def format_result(keypoints_info: np.ndarray, score_threshold: float = 0.3) -> list[PoseResult]:
        """Format raw keypoints into structured pose results.

        Converts raw keypoint arrays into structured PoseResult objects
        containing body, face, and hand keypoints with proper indexing.

        Args:
            keypoints_info: Raw keypoints array with shape (N, 134, 3) where
                N is the number of detected persons, 134 is the total number
                of keypoints, and 3 represents (x, y, score).
            score_threshold: Minimum score threshold for keypoints to be included.

        Returns:
            List of PoseResult objects for each detected person.
        """
        def create_null_keypoint(idx: int) -> Keypoint:
            return Keypoint(np.nan, np.nan, 0.0, idx)

        def format_keypoint_part(part: np.ndarray) -> list[Keypoint]:
            return [
                Keypoint(x, y, score, i) if score >= score_threshold else create_null_keypoint(i)
                for i, (x, y, score) in enumerate(part)
            ]

        pose_results = []

        for instance in keypoints_info:
            # COCO-WholeBody format: 133 keypoints total
            # Body: indices 0-16 (17 keypoints)
            # Lef Foot: indices 17-19
            # Right Foot: indices 20-22
            # Face: indices 23-90 (68 keypoints)
            # Left hand: indices 91-111 (21 keypoints)
            # Right hand: indices 112-132 (21 keypoints)
            body_keypoints = format_keypoint_part(instance[:17])
            left_foot_keypoints = format_keypoint_part(instance[17:20])
            right_foot_keypoints = format_keypoint_part(instance[20:23])
            face_keypoints = format_keypoint_part(instance[23:91])
            left_hand_keypoints = format_keypoint_part(instance[91:112])
            right_hand_keypoints = format_keypoint_part(instance[112:133])

            # Openpose face consists of 70 points in total, while RTMPose only
            # provides 68 points. Padding the last 2 points with body eye keypoints
            # left eye (body keypoint 15 in COCO format)
            if len(body_keypoints) > 15:
                face_keypoints.append(body_keypoints[15])
            else:
                face_keypoints.append(create_null_keypoint(68))
            # right eye (body keypoint 16 in COCO format)
            if len(body_keypoints) > 16:
                face_keypoints.append(body_keypoints[16])
            else:
                face_keypoints.append(create_null_keypoint(69))

            body = BodyResult(body_keypoints)
            left_hand = HandResult(left_hand_keypoints)
            right_hand = HandResult(right_hand_keypoints)
            face = FaceResult(face_keypoints)
            left_foot = FootResult(left_foot_keypoints)
            right_foot = FootResult(right_foot_keypoints)
            pose_results.append(PoseResult(body, left_hand, right_hand, face, left_foot, right_foot))

        return pose_results
