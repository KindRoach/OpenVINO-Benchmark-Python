import itertools
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import timm
import tqdm
from openvino import CompiledModel, Core, Model
from simple_parsing import choice, ArgumentParser, field
from tqdm import tqdm

from utils import MODEL_LIST, OV_MODEL_PATH_PATTERN, preprocess


@dataclass
class ExpArgs:
    model: str = choice(*MODEL_LIST, "all", alias=["-m"], default="resnet50")
    model_type: str = choice("fp32", "fp16", "int8", "all", alias=["-mt"], default="int8")
    device: str = field(alias=["-d"], default="CPU")


def parse_exp_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(ExpArgs, dest="arguments")
    return parser.parse_args(args).arguments


def load_ov_model(core: Core, model_name: str, model_type: str) -> Tuple[Model, dict]:
    cfg = timm.create_model(model_name, pretrained=True).pretrained_cfg
    model_xml = OV_MODEL_PATH_PATTERN % (model_name, model_type)
    model = core.read_model(model_xml)
    return model, cfg


def loop_seconds(seconds: int, cycle_input):
    endless_inputs = itertools.cycle([cycle_input])
    start_time = time.time()
    while time.time() - start_time < seconds:
        yield next(endless_inputs)


def get_input_shape(model: CompiledModel) -> List[int]:
    input_shape = model.input().partial_shape
    return [
        input_shape[1].min_length,
        input_shape[2].min_length,
        input_shape[3].min_length,
    ]


def get_input_dtype(model: CompiledModel) -> np.dtype:
    input_dtype = model.input().element_type
    dtype_map = {
        "f32": np.float32,
        "u8": np.uint8
    }
    return dtype_map[input_dtype.type_name]


def benchmark_model(tittle: str, model: CompiledModel, batch_size: int = 1):
    input_shape = get_input_shape(model)
    input_dtype = get_input_dtype(model)
    random_input = np.random.rand(batch_size, *input_shape).astype(input_dtype)

    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        infer_req = model.create_infer_request()
        for frame in loop_seconds(10, random_input):
            infer_req.infer(frame)
            pbar.update(1)


def benchmark_model_np_preprocess(tittle: str, model: CompiledModel, model_cfg: dict, batch_size: int = 1):
    random_input = np.random.rand(batch_size, 224, 224, 3).astype(np.uint8)

    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        infer_req = model.create_infer_request()
        for frame in loop_seconds(10, random_input):
            frame = preprocess(frame, model_cfg["input_size"], model_cfg["mean"], model_cfg["std"])
            infer_req.infer(frame)
            pbar.update(1)
