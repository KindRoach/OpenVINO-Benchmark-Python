import itertools
from threading import Lock

import numpy
from openvino.runtime import Core, CompiledModel, AsyncInferQueue
from tqdm import tqdm

from utils import MODEL_MAP, ModelMeta, OV_MODEL_PATH_PATTERN, read_input_with_time, preprocess


def benchmark_model(tittle: str, model_meta: ModelMeta, model: CompiledModel, batch_size: int = 1):
    shape = (batch_size, 1080, 1920, 3)
    random_input = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    frame = preprocess(random_input, model_meta)

    outputs = dict()
    lock = Lock()

    with tqdm(desc=tittle, unit="frame", unit_scale=batch_size) as pbar:
        def call_back(request, userdata):
            with lock:
                frame_id = userdata
                outputs[frame_id] = request.get_output_tensor().data
            pbar.update(1)

        infer_queue = AsyncInferQueue(model)
        infer_queue.set_callback(call_back)

        frames = read_input_with_time(10, model_meta, True)
        for i, _ in enumerate(frames):
            infer_queue.start_async(frame, i)

        infer_queue.wait_all()


def exp(model_meta: ModelMeta, model_type: str):
    print(f"------------------{model_meta.name}:{model_type}------------------")

    batch_size = 8

    core = Core()
    core.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})

    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, model_type)
    model = core.read_model(model_xml)

    model_single_input = core.compile_model(model)
    model_auto_batch = core.compile_model(model, f"BATCH:CPU({batch_size})")

    # change to batch size
    shapes = {}
    for input_layer in model.inputs:
        shapes[input_layer] = input_layer.partial_shape
        shapes[input_layer][0] = batch_size
    model.reshape(shapes)

    model_manual_batch = core.compile_model(model)

    benchmark_model("single input", model_meta, model_single_input)
    benchmark_model("auto batch", model_meta, model_auto_batch)
    benchmark_model("manual batch", model_meta, model_manual_batch, batch_size)


def main():
    for model, type in itertools.product(MODEL_MAP.values(), ["fp32", "fp16", "int8"]):
        exp(model, type)


if __name__ == '__main__':
    main()
