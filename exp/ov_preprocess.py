import itertools

import numpy
from openvino.preprocess import ColorFormat, ResizeAlgorithm, PrePostProcessor
from openvino.runtime import Core, Type, Layout, CompiledModel
from tqdm import tqdm

from utils import ModelMeta, read_input_with_time, MODEL_MAP, OV_MODEL_PATH_PATTERN, preprocess


def benchmark_model(tittle: str, model_meta: ModelMeta, model: CompiledModel, ov_preprocess: bool, batch_size: int = 1):
    shape = (batch_size, 1080, 1920, 3)
    random_input = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)

    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        infer_req = model.create_infer_request()
        for _ in read_input_with_time(10, model_meta, True):

            if ov_preprocess:
                frame = random_input
            else:
                frame = preprocess(random_input, model_meta)

            infer_req.infer(frame)
            infer_req.get_output_tensor()
            pbar.update(1)


def load_model(model_meta: ModelMeta, model_type: str, batch_size: int):
    core = Core()
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, model_type)
    model = core.read_model(model_xml)

    # change to dynamic shape
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = batch_size
    model.reshape(shapes)

    model_no_preprocess = core.compile_model(model)

    # build model with preprocess
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
    model_ov_preprocess = core.compile_model(model)

    return model_no_preprocess, model_ov_preprocess


def exp(model_meta: ModelMeta, model_type: str):
    print(f"------------------{model_meta.name}:{model_type}------------------")

    model_no_preprocess, model_ov_preprocess = load_model(model_meta, model_type, 1)
    benchmark_model("numpy preprocess", model_meta, model_no_preprocess, False)
    benchmark_model("openvino preprocess", model_meta, model_ov_preprocess, True)

    model_no_preprocess, model_ov_preprocess = load_model(model_meta, model_type, 8)
    benchmark_model("numpy preprocess b8", model_meta, model_no_preprocess, False, 8)
    benchmark_model("openvino preprocess b8", model_meta, model_ov_preprocess, True, 8)


def main():
    for model, type in itertools.product(MODEL_MAP.values(), ["fp32", "fp16", "int8"]):
        exp(model, type)


if __name__ == '__main__':
    main()
