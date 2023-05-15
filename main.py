import argparse
import itertools
import os
import threading

import cv2
import numpy as np
import openvino.runtime as ov
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import read_frames, IMG_SIZE

N_FRAME = 1000


def sync_infer(model: CompiledModel, video_path: str) -> np.array:
    outputs = []
    for frame in tqdm(read_frames(video_path, N_FRAME), total=N_FRAME):
        inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
        inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
        outputs.append(model(inputs))
    return outputs


def async_infer(model: CompiledModel, video_path: str, n_jobs: int) -> np.array:
    frame_count = N_FRAME
    outputs = [None] * frame_count
    with tqdm(total=frame_count) as pbar:
        def call_back(request, userdata):
            outputs[userdata] = request.get_output_tensor().data
            pbar.write(f"frame {userdata} done!")
            pbar.update(1)

        infer_queue = ov.AsyncInferQueue(model, n_jobs)
        infer_queue.set_callback(call_back)

        for i, frame in enumerate(read_frames(video_path)):
            inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            infer_queue.start_async(inputs, i)

        infer_queue.wait_all()

    return outputs


def multi_stream_infer(model: CompiledModel, video_path: str, n_stream: int) -> np.array:
    frame_count = N_FRAME // n_stream
    outputs = [[]] * n_stream
    with tqdm(total=frame_count * n_stream) as pbar:
        def infer_stream(thread_id: int):
            outputs[thread_id] = []
            for frame_id, frame in enumerate(itertools.islice(read_frames(video_path), frame_count)):
                inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
                inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
                outputs[thread_id].append(model(inputs))
                pbar.write(f"thread {thread_id} frame {frame_id} done")
                pbar.update(1)

        threads = []
        for i in range(n_stream):
            t = threading.Thread(target=infer_stream, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    return outputs


def main(args) -> None:
    ie = Core()
    devices = ie.available_devices
    for device in devices:
        device_name = ie.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    if args.infer_mode in ["async", "multi_stream"]:
        if args.infer_mode == "async":
            n_stream = args.infer_jobs
        elif args.infer_mode == "multi_stream":
            n_stream = args.n_stream

        ie.set_property("CPU", {
            "NUM_STREAMS": n_stream,
            "PERFORMANCE_HINT": "THROUGHPUT"
        })

    model_xml = f"outputs/openvino/{args.model_type}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)

    video_path = "outputs/video.mp4"
    if args.infer_mode == "async":
        async_infer(compiled_model, video_path, args.infer_jobs)
    elif args.infer_mode == "sync":
        sync_infer(compiled_model, video_path)
    elif args.infer_mode == "multi_stream":
        multi_stream_infer(compiled_model, video_path, args.n_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model_type", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-i", "--infer_mode", type=str, default="async", choices=["async", "sync", "multi_stream"])
    parser.add_argument("-j", "--infer_jobs", type=int, default=os.cpu_count())
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args)
