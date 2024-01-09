import itertools
import sys

from openvino.runtime import Core

from exp.exp_util import ExpArgs, parse_exp_args, load_ov_model, benchmark_model
from utils import MODEL_LIST


def exp(model_name: str, model_type: str):
    print(f"------------------{model_name}:{model_type}------------------")

    core = Core()
    model, _ = load_ov_model(core, model_name, model_type)
    model_static_shape_b1 = core.compile_model(model, "CPU")

    set_batch_size(model, -1)
    model_dynamic_shape = core.compile_model(model, "CPU")

    set_batch_size(model, 8)
    model_static_shape_b8 = core.compile_model(model, "CPU")

    benchmark_model("static shape b1", model_static_shape_b1)
    benchmark_model("dynamic shape b1", model_dynamic_shape)

    benchmark_model("static shape b8", model_static_shape_b8, 8)
    benchmark_model("dynamic shape b8", model_dynamic_shape, 8)


def set_batch_size(model, batch_size: int):
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = batch_size
    model.reshape(shapes)


def main(args: ExpArgs):
    models = MODEL_LIST if args.model == "all" else [args.model]
    model_types = ["fp32", "fp16", "int8"] if args.model_type == "all" else [args.model_type]
    for model, type in itertools.product(models, model_types):
        exp(model, type)


if __name__ == '__main__':
    args = parse_exp_args(sys.argv[1:])
    main(args)
