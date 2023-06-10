import time
from dataclasses import dataclass
from typing import Tuple, Callable, Dict

import cv2
import numpy
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models._api import Weights


@dataclass
class ModelMeta:
    name: str
    input_size: Tuple[int, int, int]
    input_mean: Tuple[float, float, float]
    input_std: Tuple[float, float, float]
    load_func: Callable
    weight: Weights


MODEL_MAP: Dict[str, ModelMeta] = {
    "resnet_50": ModelMeta(
        "resnet_50",
        (3, 224, 224),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        resnet50,
        ResNet50_Weights.DEFAULT
    ),
    "efficientnet_v2_l": ModelMeta(
        "efficientnet_v2_l",
        (3, 480, 480),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        efficientnet_v2_l,
        EfficientNet_V2_L_Weights.DEFAULT
    ),
}


def read_frames(video_path: str, seconds: int):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    start_time = time.time()
    while time.time() - start_time < seconds:
        success, frame = cap.read()
        if success:
            yield frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()


def read_all_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    while True:
        success, frame = cap.read()
        if success:
            yield frame
        else:
            break

    cap.release()


def preprocess(frame, model_meta: ModelMeta) -> numpy.ndarray:
    mean = 255 * numpy.array(model_meta.input_mean)
    std = 255 * numpy.array(model_meta.input_std)
    frame = cv2.resize(frame, model_meta.input_size[-2:])
    frame = frame.transpose(2, 0, 1)  # HWC to CHW
    frame = (frame - mean[:, None, None]) / std[:, None, None]
    frame = numpy.expand_dims(frame, 0)
    return frame
