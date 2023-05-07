import argparse
import os

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


def sync_infer(model: CompiledModel, video_path: str) -> None:
    for frame in tqdm(read_frames(video_path), total=total_frames(video_path)):
        inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
        inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
        model(inputs)


def async_infer(model: CompiledModel, video_path: str, n_jobs: int) -> None:
    with tqdm(total=total_frames(video_path)) as pbar:
        def call_back(request, userdata):
            pbar.update(1)

        infer_queue = ov.AsyncInferQueue(model, n_jobs)
        infer_queue.set_callback(call_back)

        for frame in read_frames(video_path):
            inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            infer_queue.start_async(inputs)

        infer_queue.wait_all()


def main(args) -> None:
    ie = Core()
    devices = ie.available_devices
    for device in devices:
        device_name = ie.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    if args.infer_mode == "async":
        ie.set_property("CPU", {
            "NUM_STREAMS": args.infer_jobs,
            "PERFORMANCE_HINT": "THROUGHPUT"
        })

    model_xml = f"outputs/openvino/{args.model_type}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)

    if args.infer_mode == "async":
        async_infer(compiled_model, "outputs/video.mp4", args.infer_jobs)
    else:
        sync_infer(compiled_model, "outputs/video.mp4")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(128)], default="CPU")
    parser.add_argument("-m", "--model_type", type=str, choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("-i", "--infer_mode", type=str, choices=["async", "sync"], default="async")
    parser.add_argument("-j", "--infer_jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args)
