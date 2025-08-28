import os
import logging
from abc import ABCMeta, abstractmethod
from typing import Any
from pathlib import Path

import numpy as np
import onnxruntime as ort
from simple_rtmw.utils import download_checkpoint


logger = logging.getLogger(__name__)


def check_mps_support():
    providers = ort.get_available_providers()
    return "MPSExecutionProvider" in providers or "CoreMLExecutionProvider" in providers


def model_exists(onnx_model: str) -> bool:
    if not os.path.exists(onnx_model):
        return False
    return True


ONNX_DEVICES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "mps": "CoreMLExecutionProvider" if check_mps_support() else "CPUExecutionProvider",
}


class BaseTool(metaclass=ABCMeta):
    def __init__(
        self,
        model_url: str,
        model_base_dir: Path,
        model_input_size: tuple,
        device: str = "cpu",
    ):
        onnx_model = download_checkpoint(model_url, model_base_dir)
        providers = ONNX_DEVICES[device]
        self.session = ort.InferenceSession(
            path_or_bytes=onnx_model, providers=[providers]
        )
        logger.info(f"load {onnx_model} with {device} device")
        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray) -> Any:
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        sess_input = {self.session.get_inputs()[0].name: input}
        sess_output = []

        for out in self.session.get_outputs():
            sess_output.append(out.name)

        outputs = self.session.run(sess_output, sess_input)
        return outputs
