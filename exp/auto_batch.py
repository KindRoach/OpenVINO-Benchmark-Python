import itertools
import sys

import numpy
import timm
from openvino.runtime import Core, CompiledModel, AsyncInferQueue
from tqdm import tqdm

from exp.exp_args import ExpArgs, parse_exp_args
from utils import OV_MODEL_PATH_PATTERN, read_input_with_time, preprocess, MODEL_LIST


def benchmark_model(tittle: str, model_cfg: dict, model: CompiledModel, batch_size: int = 1):
    shape = (batch_size, 1080, 1920, 3)
    random_input = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    frame = preprocess(random_input, model_cfg["input_size"], model_cfg["mean"], model_cfg["std"])
    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        def call_back(request, userdata):
            pbar.update(1)

        infer_queue = AsyncInferQueue(model)
        infer_queue.set_callback(call_back)

        frames = read_input_with_time(10, model_cfg["input_size"], model_cfg["mean"], model_cfg["std"], True)
        for i, _ in enumerate(frames):
            infer_queue.start_async(frame, i)

        infer_queue.wait_all()


def exp(model_cfg: dict, model_type: str):
    model_name = model_cfg['architecture']
    print(f"------------------{model_name}:{model_type}------------------")

    core = Core()
    core.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})

    model_xml = OV_MODEL_PATH_PATTERN % (model_name, model_type)
    model = core.read_model(model_xml)

    # set batch size to 1
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = 1
    model.reshape(shapes)

    model_single_input = core.compile_model(model, "CPU")

    batch_size = 8
    model_auto_batch = core.compile_model(model, f"BATCH:CPU({batch_size})")

    # change to batch size
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = batch_size
    model.reshape(shapes)

    model_manual_batch = core.compile_model(model, "CPU")

    benchmark_model("single input", model_cfg, model_single_input)
    benchmark_model("auto batch", model_cfg, model_auto_batch)
    benchmark_model("manual batch", model_cfg, model_manual_batch, batch_size)


def main(args: ExpArgs):
    models = MODEL_LIST if args.model == "all" else [args.model]
    model_types = ["fp32", "fp16", "int8"] if args.model_type == "all" else [args.model_type]
    for model, type in itertools.product(models, model_types):
        model_cfg = timm.create_model(model, pretrained=True).pretrained_cfg
        exp(model_cfg, type)


if __name__ == '__main__':
    args = parse_exp_args(sys.argv[1:])
    main(args)
