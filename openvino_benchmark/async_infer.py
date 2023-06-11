import argparse
import logging
import os
import sys
from threading import Lock
from typing import List

import openvino.runtime as ov
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import read_frames, MODEL_MAP, ModelMeta, preprocess


def async_infer(model: CompiledModel, model_meta: ModelMeta, video_path: str, runtime: int, n_jobs: int) -> list:
    outputs = dict()
    lock = Lock()
    with tqdm(unit="frame") as pbar:
        def call_back(request, userdata):
            with lock:
                outputs[userdata] = request.get_output_tensor().data
            # pbar.write(f"frame {userdata} done!")
            pbar.update(1)

        infer_queue = ov.AsyncInferQueue(model, n_jobs)
        infer_queue.set_callback(call_back)

        for i, frame in enumerate(read_frames(video_path, runtime)):
            inputs = preprocess(frame, model_meta)
            infer_queue.start_async(inputs, i)

        infer_queue.wait_all()

    return [item for key, item in sorted(outputs.items())]


def main(args) -> None:
    ie = Core()
    ie.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})

    model_meta = MODEL_MAP[args.model]
    model_xml = f"outputs/model/{model_meta.name}/openvino/{args.model_precision}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    async_infer(compiled_model, model_meta, video_path, args.run_time, args.infer_jobs)


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-n", "--infer_jobs", type=int, default=os.cpu_count())
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
