import argparse
import sys
from typing import List

from openvino.runtime import CompiledModel, Core
from tqdm import tqdm

from utils import MODEL_MAP, ModelMeta, OV_MODEL_PATH_PATTERN, read_preprocessed_frame_with_time


def sync_infer(
        model: CompiledModel,
        model_meta: ModelMeta,
        runtime: int,
        inference_only: bool) -> list:
    outputs = []
    with tqdm(unit="frame") as pbar:
        infer_req = model.create_infer_request()
        for frame in read_preprocessed_frame_with_time(runtime, model_meta, inference_only):
            infer_req.infer(frame)
            output = infer_req.get_output_tensor().data
            outputs.append(output)
            pbar.update(1)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")
    return outputs


def main(args) -> None:
    ie = Core()
    model_meta = MODEL_MAP[args.model]
    model_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, args.model_precision)
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    sync_infer(compiled_model, model_meta, args.run_time, args.inference_only)


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-t", "--run_time", type=int, default=60)
    parser.add_argument("-io", "--inference_only", action="store_true", default=False)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
