# Simple RTMW

Multi-person whole-body pose estimation using ONNX models with RTMPose and YOLOX detectors.

## Quick Start

```python
import cv2
from simple_rtmw import Wholebody
from simple_rtmw.draw import draw_annotated_image

# Initialize pipeline
model = Wholebody(device="cpu")  # Use "cuda" or "mps" for GPU

# Load image and run inference
image = cv2.imread("image.jpg")
keypoints, scores = model(image)
detection_boxes = model.det_model(image)

# Format and visualize results
keypoints_with_scores = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
pose_results = model.format_result(keypoints_with_scores)

annotated_image = draw_annotated_image(
    image,
    detection_boxes=detection_boxes,
    pose_results=pose_results
)

cv2.imwrite("output.jpg", annotated_image)
```

## Features

- **Whole-body detection**: 17 body + 68 face + 21 hand + 3 foot keypoints per person
- **Multi-platform**: CPU, CUDA, MPS (Apple Silicon) support
- **Flexible visualization**: Customizable drawing with detection boxes and pose annotations

## Configuration

```python
from simple_rtmw.draw import DrawConfig, DetectionConfig, PoseConfig

# Control what gets drawn
draw_config = DrawConfig(
    draw_detection_boxes=True,
    draw_pose_keypoints=True,
    draw_pose_skeleton=False
)

# Customize appearance
detection_config = DetectionConfig(box_color=(0, 0, 255), box_thickness=3)
pose_config = PoseConfig(keypoint_radius=5, min_score=0.5)

annotated_image = draw_annotated_image(
    image, detection_boxes, pose_results,
    draw_config=draw_config,
    detection_config=detection_config,
    pose_config=pose_config
)
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [RTMPose](https://github.com/open-mmlab/mmpose) for pose estimation models
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for detection models
- [OpenMMLab](https://openmmlab.com/) for pre-trained models and research
