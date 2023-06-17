import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import List

from openvino.runtime import Core, CompiledModel, AsyncInferQueue
from simple_parsing import choice, flag, field, ArgumentParser
from tqdm import tqdm

from utils import MODEL_MAP, ModelMeta, OV_MODEL_PATH_PATTERN, read_preprocessed_frame_with_time, cal_fps


@dataclass
class Args:
    model: str = choice(*MODEL_MAP.keys(), alias=["-m"], default="resnet_50")
    model_type: str = choice("fp32", "fp16", "int8", alias=["-mt"], default="int8")
    device: str = choice(*(["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)]), alias=["-d"], default="CPU")
    inference_only: bool = flag(alias=["-io"], default=False)
    run_mode: str = choice("sync", "async", "multi", alias=["-rm"], default="sync")
    n_stream: int = field(alias=["-n"], default=os.cpu_count())
    duration: int = field(alias=["-t"], default=60)


def sync_infer(args: Args, model: CompiledModel, model_meta: ModelMeta) -> list:
    outputs = []
    with tqdm(unit="frame") as pbar:
        infer_req = model.create_infer_request()
        for frame in read_preprocessed_frame_with_time(args.duration, model_meta, args.inference_only):
            infer_req.infer(frame)
            output = infer_req.get_output_tensor().data
            outputs.append(output)
            pbar.update(1)

    cal_fps(pbar)
    return outputs


def async_infer(args: Args, model: CompiledModel, model_meta: ModelMeta) -> list:
    outputs = dict()
    lock = Lock()
    with tqdm(unit="frame") as pbar:
        def call_back(request, userdata):
            with lock:
                frame_id = userdata
                outputs[frame_id] = request.get_output_tensor().data
            pbar.update(1)

        infer_queue = AsyncInferQueue(model)
        infer_queue.set_callback(call_back)

        frames = read_preprocessed_frame_with_time(args.duration, model_meta, args.inference_only)
        for i, frame in enumerate(frames):
            infer_queue.start_async(frame, i)

        infer_queue.wait_all()

    return [item for key, item in sorted(outputs.items())]


def multi_infer(args: Args, model: CompiledModel, model_meta: ModelMeta) -> list:
    with tqdm(unit="frame") as pbar:
        def infer_stream(thread_id: int):
            outputs = []
            infer_req = model.create_infer_request()
            frames = read_preprocessed_frame_with_time(args.duration, model_meta, args.inference_only)
            for frame_id, frame in enumerate(frames):
                infer_req.start_async(frame)
                infer_req.wait()
                output = infer_req.get_output_tensor().data
                outputs.append(output)
                pbar.update(1)
            return outputs

        with ThreadPoolExecutor(args.n_stream) as pool:
            tasks = []
            for tid in range(args.n_stream):
                tasks.append(pool.submit(infer_stream, tid))
            outputs = [task.result() for task in tasks]

    cal_fps(pbar)
    return outputs


def main(args: Args) -> None:
    ie = Core()
    throughput_mode = "THROUGHPUT" if args.run_mode in ["async", "multi"] else "LATENCY"
    ie.set_property("CPU", {"PERFORMANCE_HINT": throughput_mode})
    model_meta = MODEL_MAP[args.model]
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, args.model_type)
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    globals()[f"{args.run_mode}_infer"](args, compiled_model, model_meta)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
