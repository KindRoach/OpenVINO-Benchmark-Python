import time
from dataclasses import dataclass
from typing import Tuple, Callable, Dict

import cv2
import numpy
from openvino.preprocess import ColorFormat, ResizeAlgorithm, PrePostProcessor
from openvino.runtime import Core, Type, Layout
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models._api import Weights

OV_MODEL_PATH_PATTERN = "outputs/model/%s/%s/model.xml"
VIDEO_PATH = "outputs/video.mp4"


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


def read_endless_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened()

    while True:
        success, frame = cap.read()
        if success:
            yield frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()


def read_all_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose(2, 0, 1)  # HWC to CHW
    frame = (frame - mean[:, None, None]) / std[:, None, None]
    frame = numpy.expand_dims(frame, 0)
    return frame


def read_frames_with_time(seconds: int):
    endless_frames = iter(read_endless_frames())

    start_time = time.time()
    while time.time() - start_time < seconds:
        yield next(endless_frames)


def read_input_with_time(seconds: int, model_meta: ModelMeta, inference_only: bool, preprocess_frame: bool):
    if preprocess_frame:
        shape = (1, *model_meta.input_size)
        random_input = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    else:
        shape = (1, 1080, 1920, 3)
        random_input = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)

    endless_frames = iter(read_endless_frames())

    start_time = time.time()
    while time.time() - start_time < seconds:
        if inference_only:
            yield random_input
        else:
            frame = next(endless_frames)
            if preprocess_frame:
                frame = preprocess(frame, model_meta)
            else:
                frame = numpy.expand_dims(frame, 0)
            yield frame


def load_model(core: Core, model_meta: ModelMeta, model_type: str, ov_preprocess: bool):
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, model_type)
    model = core.read_model(model_xml)
    if ov_preprocess:
        ppp = PrePostProcessor(model)

        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_spatial_dynamic_shape() \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)

        ppp.input().model().set_layout(Layout('NCHW'))

        mean = 255 * numpy.array(model_meta.input_mean)
        scale = 255 * numpy.array(model_meta.input_std)

        ppp.input().preprocess() \
            .convert_element_type(Type.f32) \
            .convert_color(ColorFormat.RGB) \
            .resize(ResizeAlgorithm.RESIZE_LINEAR) \
            .mean(mean) \
            .scale(scale)

        model = ppp.build()

    return core.compile_model(model)


def cal_fps(pbar):
    frames = pbar.format_dict["n"]
    seconds = pbar.format_dict["elapsed"]
    print(f"fps: {frames / seconds:.2f}")
