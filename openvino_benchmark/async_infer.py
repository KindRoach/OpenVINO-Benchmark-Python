import argparse
import sys
from threading import Lock
from typing import List

import openvino.runtime as ov
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import MODEL_MAP, ModelMeta, OV_MODEL_PATH_PATTERN, read_preprocessed_frame_with_time


def async_infer(
        model: CompiledModel,
        model_meta: ModelMeta,
        runtime: int,
        inference_only) -> list:
    outputs = dict()
    lock = Lock()
    with tqdm(unit="frame") as pbar:
        def call_back(request, userdata):
            with lock:
                frame_id = userdata
                outputs[frame_id] = request.get_output_tensor().data
            # pbar.write(f"frame {userdata} done!")
            pbar.update(1)

        infer_queue = ov.AsyncInferQueue(model)
        infer_queue.set_callback(call_back)

        frames = read_preprocessed_frame_with_time(runtime, model_meta, inference_only)
        for i, frame in enumerate(frames):
            infer_queue.start_async(frame, i)

        infer_queue.wait_all()

    return [item for key, item in sorted(outputs.items())]


def main(args) -> None:
    ie = Core()
    ie.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})

    model_meta = MODEL_MAP[args.model]
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, args.model_precision)
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    async_infer(compiled_model, model_meta, args.run_time, args.inference_only)


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-io", "--inference_only", action="store_true", default=False)
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
