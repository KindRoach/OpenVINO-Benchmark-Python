import time
from dataclasses import dataclass
from typing import Tuple, Callable, Dict

import cv2
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models._api import Weights


@dataclass
class ModelMeta:
    name: str
    input_size: Tuple[int, int, int]
    load_func: Callable
    weight: Weights


MODEL_MAP: Dict[str, ModelMeta] = {
    "resnet_50": ModelMeta(
        "resnet_50",
        (3, 224, 224),
        resnet50,
        ResNet50_Weights.DEFAULT
    ),
    "efficientnet_v2_l": ModelMeta(
        "efficientnet_v2_l",
        (3, 480, 480),
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
