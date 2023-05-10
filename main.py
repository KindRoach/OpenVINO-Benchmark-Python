import argparse
import itertools
import os
import threading

import cv2
import numpy as np
import openvino.runtime as ov
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

IMG_SIZE = 480


def total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def read_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    success, frame = cap.read()
    while success:
        yield frame
        success, frame = cap.read()
    cap.release()


def sync_infer(model: CompiledModel, video_path: str) -> np.array:
    outputs = []
    for frame in tqdm(read_frames(video_path), total=total_frames(video_path)):
        inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
        inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
        outputs.append(model(inputs))
    return outputs


def async_infer(model: CompiledModel, video_path: str, n_jobs: int) -> np.array:
    frame_count = total_frames(video_path)
    outputs = [None] * frame_count
    with tqdm(total=frame_count) as pbar:
        def call_back(request, userdata):
            outputs[userdata] = request.get_output_tensor().data
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
    frame_count = total_frames(video_path) // n_stream
    outputs = [None] * n_stream
    threads = []
    with tqdm(total=frame_count * n_stream) as pbar:
        def infer_stream(model: CompiledModel, video_path: str, idx: int):
            outputs[idx] = []
            for i, frame in enumerate(itertools.islice(read_frames(video_path), frame_count)):
                inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
                inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
                outputs[idx].append(model(inputs))
                pbar.write(f"thread {idx} frame {i} done")
                pbar.update(1)

        for i in range(n_stream):
            t = threading.Thread(target=infer_stream, args=(model, video_path, i))
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

    if args.infer_mode == "async":
        async_infer(compiled_model, "outputs/video.mp4", args.infer_jobs)
    elif args.infer_mode == "sync":
        sync_infer(compiled_model, "outputs/video.mp4")
    elif args.infer_mode == "multi_stream":
        multi_stream_infer(compiled_model, "outputs/video.mp4", args.n_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(128)])
    parser.add_argument("-m", "--model_type", type=str, default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-i", "--infer_mode", type=str, default="async", choices=["async", "sync", "multi_stream"])
    parser.add_argument("-j", "--infer_jobs", type=int, default=os.cpu_count())
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args)
