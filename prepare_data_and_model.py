import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

import nncf
import numpy
import openvino as ov
import timm
import torch
from simple_parsing import choice, ArgumentParser
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from utils import read_all_frames, preprocess, OV_MODEL_PATH_PATTERN, TEST_VIDEO_PATH, TEST_IMAGE_PATH, MODEL_LIST


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


def convert_torch_to_openvino(model: Module) -> None:
    logging.info("Converting Model to OpenVINO...")

    cfg = model.pretrained_cfg
    input_shape = [-1, *cfg["input_size"]]
    ov_model = ov.convert_model(model, input=[input_shape])

    model_fp32_xml = OV_MODEL_PATH_PATTERN % (cfg["architecture"], "fp32")
    Path(model_fp32_xml).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, model_fp32_xml, compress_to_fp16=False)

    model_fp16_xml = OV_MODEL_PATH_PATTERN % (cfg["architecture"], "fp16")
    ov.save_model(ov_model, model_fp16_xml, compress_to_fp16=True)


def quantization(model: Module) -> None:
    cfg = model.pretrained_cfg

    model_name = cfg['architecture']
    logging.info(f"{model_name} Model Quantization...")

    frames = []
    for frame in read_all_frames():
        frame = preprocess(frame, cfg["input_size"], cfg["mean"], cfg["std"])[0]
        frames.append(frame)

    frames = torch.tensor(numpy.array(frames))
    dataloader = DataLoader(TensorDataset(frames))
    dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())

    model_fp32_xml = OV_MODEL_PATH_PATTERN % (model_name, "fp32")
    model_int8_xml = OV_MODEL_PATH_PATTERN % (model_name, "int8")
    model_fp32 = ov.Core().read_model(model_fp32_xml)
    model_int8 = nncf.quantize(model_fp32, dataset, subset_size=len(dataloader))
    ov.serialize(model_int8, model_int8_xml)


@dataclass
class Args:
    model: str = choice(*MODEL_LIST, "all", alias=["-m"], default="resnet50")


def main(args: Args) -> None:
    download_video_and_image()
    models = MODEL_LIST if args.model == "all" else [args.model]

    for model_name in models:
        model = timm.create_model(model_name, pretrained=True)
        convert_torch_to_openvino(model)
        quantization(model)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
