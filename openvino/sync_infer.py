import argparse

import cv2
import numpy as np
from openvino.runtime import CompiledModel, Core
from tqdm import tqdm

from main import N_FRAME
from utils import read_frames, IMG_SIZE


def sync_infer(model: CompiledModel, video_path: str) -> np.array:
    outputs = []
    for frame in tqdm(read_frames(video_path, N_FRAME), total=N_FRAME):
        inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
        inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
        outputs.append(model(inputs))
    return outputs


def main(args) -> None:
    ie = Core()
    model_xml = f"outputs/openvino/{args.model_type}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    sync_infer(compiled_model, video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model_type", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()

    main(args)
