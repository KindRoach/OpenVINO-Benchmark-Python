import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

import nncf
import numpy
import openvino.runtime as ov
import torch
from openvino.tools import mo
from simple_parsing import choice, ArgumentParser
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from utils import MODEL_MAP, ModelMeta, read_all_frames, preprocess, OV_MODEL_PATH_PATTERN, TEST_VIDEO_PATH, \
    TEST_IMAGE_PATH


def download_file(url: str, target_path: str) -> None:
    if not Path(target_path).exists():
        logging.info(f"Downloading to {target_path} ...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, target_path)


def download_video_and_image() -> None:
    download_file(
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4",
        TEST_VIDEO_PATH
    )

    download_file(
        "https://storage.openvinotoolkit.org/data/test_data/images/dog.jpg",
        TEST_IMAGE_PATH
    )


def download_model(model: ModelMeta) -> Module:
    logging.info("Downloading Model...")

    weight = model.weight
    load_func = model.load_func

    model = load_func(weights=weight)
    model.eval()

    return model


def convert_torch_to_openvino(model_meta: ModelMeta, model: Module) -> None:
    logging.info("Converting Model to OpenVINO...")

    model_fp32 = mo.convert_model(model, input_shape=(1, *model_meta.input_size))
    model_fp32_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, "fp32")
    Path(model_fp32_xml).parent.mkdir(parents=True, exist_ok=True)
    ov.serialize(model_fp32, model_fp32_xml)

    model_fp16 = mo.convert_model(model, input_shape=(1, *model_meta.input_size), compress_to_fp16=True)
    model_fp16_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, "fp16")
    ov.serialize(model_fp16, model_fp16_xml)


def quantization(model_meta: ModelMeta) -> None:
    logging.info(f"{model_meta.name} Model Quantization...")

    frames = []
    for frame in read_all_frames():
        frame = preprocess(frame, model_meta)[0]
        frames.append(frame)

    frames = torch.tensor(numpy.array(frames))
    dataloader = DataLoader(TensorDataset(frames), batch_size=1)
    dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())

    model_fp32_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, "fp32")
    model_int8_xml = OV_MODEL_PATH_PATTERN % (model_meta.name, "int8")
    model_fp32 = ov.Core().read_model(model_fp32_xml)
    model_int8 = nncf.quantize(model_fp32, dataset)
    ov.serialize(model_int8, model_int8_xml)


@dataclass
class Args:
    model: str = choice(*MODEL_MAP.keys(), "all", alias=["-m"], default="all")


def main(args: Args) -> None:
    download_video_and_image()
    models = MODEL_MAP.keys() if args.model == "all" else [args.model]

    for model_name in models:
        model_meta = MODEL_MAP[model_name]
        model = download_model(model_meta)
        convert_torch_to_openvino(model_meta, model)
        quantization(model_meta)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
