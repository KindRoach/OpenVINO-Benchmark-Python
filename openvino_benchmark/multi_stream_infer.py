import argparse
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import read_frames, MODEL_MAP, ModelMeta


def multi_stream_infer(model: CompiledModel, model_meta: ModelMeta, video_path: str, runtime: int, n_stream: int) -> list:
    with tqdm(unit="frame") as pbar:
        def infer_stream(thread_id: int):
            output = []
            for frame_id, frame in enumerate(read_frames(video_path, runtime)):
                inputs = cv2.resize(src=frame, dsize=model_meta.input_size[-2:])
                inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
                output.append(model(inputs))
                # pbar.write(f"thread {thread_id} frame {frame_id} done")
                pbar.update(1)
            return output

        with ThreadPoolExecutor(n_stream) as pool:
            tasks = []
            for tid in range(n_stream):
                tasks.append(pool.submit(infer_stream, tid))
            outputs = [task.result() for task in tasks]

    return outputs


def main(args) -> None:
    ie = Core()
    ie.set_property("CPU", {
        "NUM_STREAMS": args.n_stream,
        "PERFORMANCE_HINT": "THROUGHPUT"
    })

    model_meta = MODEL_MAP[args.model]
    model_xml = f"outputs/model/{model_meta.name}/openvino/{args.model_precision}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    multi_stream_infer(compiled_model, model_meta, video_path, args.run_time, args.n_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    parser.add_argument("-t", "--run_time", type=int, default=60)
    args = parser.parse_args()

    main(args)
