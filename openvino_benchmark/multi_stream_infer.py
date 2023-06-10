import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import read_frames, MODEL_MAP, ModelMeta, preprocess


def multi_stream_infer(model: CompiledModel, model_meta: ModelMeta, video_path: str, runtime: int,
                       n_stream: int) -> list:
    with tqdm(unit="frame") as pbar:
        def infer_stream(thread_id: int):
            outputs = []
            infer_req = model.create_infer_request()
            for frame_id, frame in enumerate(read_frames(video_path, runtime)):
                input_frame = preprocess(frame, model_meta)
                infer_req.infer(input_frame)
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
    model_xml = f"outputs/model/{model_meta.name}/openvino/{args.model_precision}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    multi_stream_infer(compiled_model, model_meta, video_path, args.run_time, args.n_stream)


def parse_ages(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_ages(sys.argv[1:])
    main(args)
