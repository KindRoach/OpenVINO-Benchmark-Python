import itertools
import sys

import numpy
import timm
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from exp.exp_args import ExpArgs, parse_exp_args
from utils import OV_MODEL_PATH_PATTERN, read_input_with_time, MODEL_LIST

core = Core()


def benchmark_model(tittle: str, model_cfg: dict, model: CompiledModel, batch_size: int = 1):
    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        infer_req = model.create_infer_request()
        for frame in read_input_with_time(10, model_cfg["input_size"], model_cfg["mean"], model_cfg["std"], True):
            frame = numpy.repeat(frame, batch_size, axis=0)
            infer_req.infer(frame)
            infer_req.get_output_tensor()
            pbar.update(1)


def exp(model_cfg: dict, model_type: str):
    model_name = model_cfg['architecture']
    print(f"------------------{model_name}:{model_type}------------------")
    model_xml = OV_MODEL_PATH_PATTERN % (model_name, model_type)

    core = Core()
    model = core.read_model(model_xml)
    model_static_shape = core.compile_model(model, "CPU")

    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = -1
    model.reshape(shapes)
    model_dynamic_shape = core.compile_model(model, "CPU")

    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = 8
    model.reshape(shapes)
    model_static_shape_b8 = core.compile_model(model, "CPU")

    benchmark_model("static shape", model_cfg, model_static_shape)
    benchmark_model("dynamic shape b1", model_cfg, model_dynamic_shape)

    benchmark_model("static shape b8", model_cfg, model_static_shape_b8, 8)
    benchmark_model("dynamic shape b8", model_cfg, model_dynamic_shape, 8)


def main(args: ExpArgs):
    models = MODEL_LIST if args.model == "all" else [args.model]
    model_types = ["fp32", "fp16", "int8"] if args.model_type == "all" else [args.model_type]
    for model, type in itertools.product(models, model_types):
        model_cfg = timm.create_model(model, pretrained=True).pretrained_cfg
        exp(model_cfg, type)


if __name__ == '__main__':
    args = parse_exp_args(sys.argv[1:])
    main(args)
