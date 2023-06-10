import argparse
import sys
from typing import List

from openvino.runtime import CompiledModel, Core
from tqdm import tqdm

from utils import read_frames, MODEL_MAP, ModelMeta, preprocess


def sync_infer(model: CompiledModel, model_meta: ModelMeta, video_path: str, runtime: int) -> list:
    outputs = []
    with tqdm(unit="frame") as pbar:
        for frame in read_frames(video_path, runtime):
            inputs = preprocess(frame, model_meta)
            outputs.append(model(inputs)[model.output(0)])
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


def parse_ages(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_ages(sys.argv[1:])
    main(args)
