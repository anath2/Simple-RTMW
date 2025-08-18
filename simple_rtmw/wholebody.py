import logging
import numpy as np

from simple_rtmw.detection import Detector
from simple_rtmw.pose import PoseEstimator
from simple_rtmw.types import BodyResult, Keypoint, PoseResult


logger = logging.getLogger(__name__)

CONFIG = {
    'detector': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
    'detector_input_size': (640, 640),
    'pose_estimator': 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',  # noqa
    'pose_estimator_input_size': (288, 384),
    'backend': 'onnxruntime',
    'device': 'mps',
}


class Wholebody:

    def __init__(self,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose: str = None,
        pose_input_size: tuple = (288, 384),
        backend: str = 'onnxruntime',
        device: str = 'cpu',
    ):

        self.det_model = Detector(det,
                               model_input_size=det_input_size,
                               backend=backend,
                               device=device)
        self.pose_model = PoseEstimator(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=False,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores

    @staticmethod
    def format_result(keypoints_info: np.ndarray) -> list[PoseResult]:

        def format_keypoint_part(
                part: np.ndarray) -> list[Keypoint | None] | None:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (None if all(keypoint is None
                                for keypoint in keypoints) else keypoints)

        def total_score(
                keypoints: list[Keypoint | None] | None) -> float:
            return (sum(
                keypoint.score for keypoint in keypoints
                if keypoint is not None) if keypoints is not None else 0.0)

        pose_results = []

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(
                instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while RTMPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(body_keypoints, total_score(body_keypoints),
                              len(body_keypoints))
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results