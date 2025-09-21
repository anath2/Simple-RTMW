from pathlib import Path

import numpy as np

from simple_rtmw.base import BaseTool
from simple_rtmw.pose.post import get_simcc_maximum
from simple_rtmw.pose.pre import bbox_xyxy2cs, top_down_affine


class RTMPose(BaseTool):
    def __init__(
        self,
        model_url: str,
        model_base_dir: Path,
        device: str,
        model_input_size: tuple = (288, 384),    # rtmw-xl input size defined in the model
        mean: tuple = (123.675, 116.28, 103.53),  # imagenet convention mean
        std: tuple = (58.395, 57.12, 57.375),  # imagenet convention std
    ):
        super().__init__(
            model_url,
            model_base_dir,
            model_input_size,
            device=device,
        )

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.model_input_size = np.array(model_input_size)

    def __call__(self, image: np.ndarray, bboxes: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if bboxes is None:  # if bboxes is None, use the whole image
            bboxes = [np.array([0, 0, image.shape[1], image.shape[0]])]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)
        return keypoints, scores

    def preprocess(
            self,
            img: np.ndarray,
            bbox: np.ndarray,
            bbox_padding_factor: float = 1.25,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Do preprocessing for RTMPose model inference.

        Args:
            img: Input image in shape.
            bbox:  xyxy-format bounding box of target.
            bbox_padding_factor: float = 1.25

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=bbox_padding_factor)

        # do affine transformation
        resized_img, scale = top_down_affine(
            self.model_input_size,
            scale,
            center,
            img,
        )

        # normalize image
        if self.mean is not None:
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
        self,
        outputs: list[np.ndarray],
        center: np.ndarray,
        scale: np.ndarray,
        simcc_split_ratio: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs: Output of RTMPose model.
            center: Center of the image
            scale: Scale of image
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            - keypoints: Rescaled keypoints.
            - scores: Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores
