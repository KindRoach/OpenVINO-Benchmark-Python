import argparse

import cv2
import numpy as np
from openvino.runtime import CompiledModel, Core
from tqdm import tqdm

from utils import read_frames, MODEL_MAP, ModelMeta


def sync_infer(model: CompiledModel, model_meta: ModelMeta, video_path: str, runtime: int) -> list:
    outputs = []
    with tqdm(unit="frame") as pbar:
        for frame in read_frames(video_path, runtime):
            inputs = cv2.resize(src=frame, dsize=model_meta.input_size[-2:])
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            outputs.append(model(inputs))
            pbar.update(1)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")
    return outputs


def main(args) -> None:
    ie = Core()
    model_meta = MODEL_MAP[args.model]
    model_xml = f"outputs/model/{model_meta.name}/openvino/{args.model_precision}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    sync_infer(compiled_model, model_meta, video_path, args.run_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-t", "--run_time", type=int, default=60)
    args = parser.parse_args()

    main(args)
