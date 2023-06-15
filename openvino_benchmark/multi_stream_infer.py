import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import MODEL_MAP, ModelMeta, OV_MODEL_PATH_PATTERN, read_preprocessed_frame_with_time


def multi_stream_infer(
        model: CompiledModel,
        model_meta: ModelMeta,
        runtime: int,
        inference_only: bool,
        n_stream: int) -> list:
    with tqdm(unit="frame") as pbar:
        def infer_stream(thread_id: int):
            outputs = []
            infer_req = model.create_infer_request()
            frames = read_preprocessed_frame_with_time(runtime, model_meta, inference_only)
            for frame_id, frame in enumerate(frames):
                infer_req.infer(frame)
                output = infer_req.get_output_tensor().data
                outputs.append(output)
                # pbar.write(f"thread {thread_id} frame {frame_id} done")
                pbar.update(1)
            return outputs

        with ThreadPoolExecutor(n_stream) as pool:
            tasks = []
            for tid in range(n_stream):
                tasks.append(pool.submit(infer_stream, tid))
            outputs = [task.result() for task in tasks]

    return outputs


def main(args) -> None:
    ie = Core()
    ie.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})

    model_meta = MODEL_MAP[args.model]
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, args.model_precision)
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    multi_stream_infer(compiled_model, model_meta, args.run_time, args.inference_only, args.n_stream)


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-io", "--inference_only", action="store_true", default=False)
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
