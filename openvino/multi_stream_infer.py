import argparse
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import read_frames, IMG_SIZE

N_FRAME = 1000


def multi_stream_infer(model: CompiledModel, video_path: str, n_stream: int) -> np.array:
    frame_count = N_FRAME // n_stream
    with tqdm(total=frame_count * n_stream) as pbar:
        def infer_stream(thread_id: int):
            output = []
            for frame_id, frame in enumerate(itertools.islice(read_frames(video_path, N_FRAME), frame_count)):
                inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
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

    model_xml = f"outputs/openvino/{args.model_type}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    multi_stream_infer(compiled_model, video_path, args.n_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model_type", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args)
