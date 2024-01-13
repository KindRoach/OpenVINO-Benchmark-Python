import itertools
import sys

import numpy
from openvino import Model
from openvino.preprocess import ColorFormat, PrePostProcessor
from openvino.runtime import Core, Type, Layout

from exp.dynamic_shape import set_batch_size
from exp.exp_util import ExpArgs, parse_exp_args, load_ov_model, benchmark_model, benchmark_model_np_preprocess
from utils import MODEL_LIST

core = Core()


def exp(model_name: str, model_type: str, device: str):
    print(f"------------------{model_name}:{model_type}------------------")

    model_no_preprocess, model_cfg = load_ov_model(core, model_name, model_type)
    set_batch_size(model_no_preprocess, 1)
    model_ov_preprocess = build_ppp_model(model_name, model_type)

    mnp = core.compile_model(model_no_preprocess, device)
    movp = core.compile_model(model_ov_preprocess, device)
    benchmark_model_np_preprocess("numpy preprocess b1", mnp, model_cfg)
    benchmark_model("openvino preprocess b1", movp)

    batch_size = 8
    set_batch_size(model_no_preprocess, batch_size)
    set_batch_size(model_ov_preprocess, batch_size)
    mnp = core.compile_model(model_no_preprocess, device)
    movp = core.compile_model(model_ov_preprocess, device)
    benchmark_model_np_preprocess("numpy preprocess b8", mnp, model_cfg, batch_size)
    benchmark_model("openvino preprocess b8", movp, batch_size)


def build_ppp_model(model_name: str, model_type: str) -> Model:
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

    return ppp.build()


def main(args: ExpArgs):
    models = MODEL_LIST if args.model == "all" else [args.model]
    model_types = ["fp32", "fp16", "int8"] if args.model_type == "all" else [args.model_type]
    for model, type in itertools.product(models, model_types):
        exp(model, type, args.device)


if __name__ == '__main__':
    main(parse_exp_args(sys.argv[1:]))
