from pathlib import Path

import cv2
import numpy as np

from simple_rtmw.base import BaseTool


class YOLOX(BaseTool):
    def __init__(
        self,
        model_url: str,
        model_base_dir: Path,
        model_input_size: tuple[int, int] = (640, 640),
        nms_thr: float = 0.45,
        score_thr: float = 0.7,
        device: str = 'cpu',
    ):
        super().__init__(
            model_url,
            model_base_dir,
            model_input_size,
            device=device,
        )
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        return self.postprocess(outputs, ratio)

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Preprocessing for inference.

        Args:
            img: Input image in HWC format (grayscale, RGB, BGR, or RGBA).

        Returns:
            - padded_img: Letterboxed image resized to model_input_size
              with aspect ratio preserved. Gray padding (value 114) fills remaining area.
              Shape: (model_input_size[0], model_input_size[1], 3).
            - ratio: Scale factor used for resizing. Used in postprocessing
              to map bounding boxes back to original image coordinates.
        """
        if img.ndim == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # rgba image
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:  # rgb image
            pass

        padded_img = np.ones(
            (self.model_input_size[0], self.model_input_size[1], 3),
            dtype=np.uint8,
        ) * 114

        ratio = min(
            self.model_input_size[0] / img.shape[0],
            self.model_input_size[1] / img.shape[1],
        )

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img
        return padded_img, ratio

    def postprocess(
        self,
        outputs: list[np.ndarray],
        ratio: float = 1.,
    ) -> np.ndarray:
        """Postprocesses YOLOX model outputs to produce final bounding boxes in original image coordinates.
        Assumes batch size of 1.

        Args:
            outputs: Raw outputs from the YOLOX model.
                Expected shape: (1, N, 5), where N is the number of detections,
                and each detection is [x1, y1, x2, y2, score].
            ratio: The scaling ratio used during preprocessing.
                Used to map bounding boxes back to the original image size. Default is 1.0.

        Returns:
            Array of final bounding boxes after thresholding and rescaling.
            Shape: (num_boxes, 4). Each box is [x1, y1, x2, y2] in original image coordinates.
        """
        # Split boxes and scores
        detections = outputs[0]
        final_boxes = detections[:, :4]
        final_scores = detections[:, 4]

        final_boxes_scaled = final_boxes / ratio
        return final_boxes_scaled[final_scores > self.score_thr].astype(np.int32)
