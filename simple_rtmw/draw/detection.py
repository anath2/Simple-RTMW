"""Drawing functions for detection results."""

import cv2
import numpy as np

from simple_rtmw.draw.config import DetectionConfig


def draw_detection_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    config: DetectionConfig | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw detection bounding boxes on image.

    Args:
        image: Input image in BGR format (H, W, 3)
        boxes: Detection boxes in format (N, 4) where each box is [x1, y1, x2, y2]
        config: Detection configuration, uses default if None
        copy: If True, draw on a copy of the image. If False, draw in-place

    Returns:
        Image with detection boxes drawn
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be in BGR format (H, W, 3)")

    if len(boxes) == 0:
        # No boxes to draw, return image as-is
        return image.copy() if copy else image

    if boxes.shape[1] != 4:
        raise ValueError("Boxes must have shape (N, 4)")

    if config is None:
        config = DetectionConfig()

    result_image = image.copy() if copy else image

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            config.box_color,
            config.box_thickness,
        )

    return result_image
