import itertools

import numpy
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import ModelMeta, OV_MODEL_PATH_PATTERN, MODEL_MAP, read_input_with_time

core = Core()


def benchmark_model(tittle: str, model_meta: ModelMeta, model: CompiledModel, batch_size: int = 1):
    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        infer_req = model.create_infer_request()
        for frame in read_input_with_time(10, model_meta, True):
            frame = numpy.repeat(frame, batch_size, axis=0)
            infer_req.infer(frame)
            infer_req.get_output_tensor()
            pbar.update(1)


def exp(model_meta: ModelMeta, model_type: str):
    core = Core()
    print(f"------------------{model_meta.name}:{model_type}------------------")
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, model_type)
    model = core.read_model(model_xml)
    model_static_shape = core.compile_model(model)

    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = -1
    model.reshape(shapes)
    model_dynamic_shape = core.compile_model(model)

    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = 8
    model.reshape(shapes)
    model_static_shape_b8 = core.compile_model(model)

    benchmark_model("static shape", model_meta, model_static_shape)
    benchmark_model("dynamic shape b1", model_meta, model_dynamic_shape)

    benchmark_model("static shape b8", model_meta, model_static_shape_b8, 8)
    benchmark_model("dynamic shape b8", model_meta, model_dynamic_shape, 8)


def main():
    for model, type in itertools.product(MODEL_MAP.values(), ["fp32", "fp16", "int8"]):
        exp(model, type)


if __name__ == '__main__':
    main()
