import itertools
import sys

import numpy
from openvino import Model
from openvino.preprocess import ColorFormat, PrePostProcessor
from openvino.runtime import Core, Type, Layout

from exp.dynamic_shape import set_batch_size
from exp.exp_util import ExpArgs, parse_exp_args, load_ov_model, benchmark_model
from utils import MODEL_LIST

core = Core()


def exp(model_name: str, model_type: str, device: str):
    print(f"------------------{model_name}:{model_type}------------------")

    model_no_ppp, model_cfg = load_ov_model(core, model_name, model_type)
    set_batch_size(model_no_ppp, 1)
    model_complex_ppp = build_complex_ppp_model(model_name, model_type)
    model_simple_ppp = build_simple_ppp_model(model_name, model_type)

    mnp = core.compile_model(model_no_ppp, device)
    mcp = core.compile_model(model_complex_ppp, device)
    msp = core.compile_model(model_simple_ppp, device)
    benchmark_model("No openvino preprocess b1", mnp)
    benchmark_model("Complex openvino preprocess b1", mcp)
    benchmark_model("Simple openvino preprocess b1", msp)

    batch_size = 8
    set_batch_size(model_no_ppp, batch_size)
    model_complex_ppp = build_complex_ppp_model(model_name, model_type)
    model_simple_ppp = build_simple_ppp_model(model_name, model_type)

    mnp = core.compile_model(model_no_ppp, device)
    mcp = core.compile_model(model_complex_ppp, device)
    msp = core.compile_model(model_simple_ppp, device)
    benchmark_model("No openvino preprocess b8", mnp, 8)
    benchmark_model("Complex openvino preprocess b8", mcp, 8)
    benchmark_model("Simple openvino preprocess b8", msp, 8)


def build_complex_ppp_model(model_name: str, model_type: str) -> Model:
    model, model_cfg = load_ov_model(core, model_name, model_type)

    # build model with preprocessing
    ppp = PrePostProcessor(model)
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.BGR)

    ppp.input().model().set_layout(Layout('NCHW'))

    mean = 255 * numpy.array(model_cfg["mean"])
    scale = 255 * numpy.array(model_cfg["std"])
    ppp.input().preprocess() \
        .convert_element_type(Type.f32) \
        .convert_color(ColorFormat.RGB) \
        .mean(mean) \
        .scale(scale)

    print("Complex preprocess:")
    print(ppp)

    return ppp.build()


def build_simple_ppp_model(model_name: str, model_type: str) -> Model:
    model, model_cfg = load_ov_model(core, model_name, model_type)

    # build model with preprocessing
    ppp = PrePostProcessor(model)
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))

    ppp.input().model().set_layout(Layout('NCHW'))

    print("Simple preprocess:")
    print(ppp)

    return ppp.build()


def main(args: ExpArgs):
    models = MODEL_LIST if args.model == "all" else [args.model]
    model_types = ["fp32", "fp16", "int8"] if args.model_type == "all" else [args.model_type]
    for model, type in itertools.product(models, model_types):
        exp(model, type, args.device)


if __name__ == '__main__':
    main(parse_exp_args(sys.argv[1:]))
