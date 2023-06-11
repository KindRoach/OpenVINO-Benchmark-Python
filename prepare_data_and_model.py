import logging
import subprocess
import urllib.request
from pathlib import Path

import nncf
import numpy
import openvino.runtime as ov
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from utils import MODEL_MAP, ModelMeta, read_all_frames, preprocess


def download_video() -> None:
    video_path = "outputs/video.mp4"
    if not Path(video_path).exists():
        logging.info("Downloading Video...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4"
        urllib.request.urlretrieve(video_url, video_path)


def download_model(model: ModelMeta) -> Module:
    logging.info("Downloading Model...")

    weight = model.weight
    load_func = model.load_func

    model = load_func(weights=weight)
    model.eval()

    return model


def convert_torch_to_openvino(model_meta: ModelMeta, model: Module) -> None:
    logging.info("Converting Model to OpenVINO...")

    # convert to onnx
    onnx_path = f"outputs/model/{model_meta.name}/model.onnx"
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, *model_meta.input_size)
    torch.onnx.export(model, (dummy_input,), onnx_path)

    # convert to openvino
    openvino_path = f"outputs/model/{model_meta.name}/openvino"
    subprocess.run([
        "mo",
        "--input_model", onnx_path,
        "--output_dir", f"{openvino_path}/fp32",
        "--log_level", "ERROR"
    ])
    subprocess.run([
        "mo",
        "--compress_to_fp16",
        "--input_model", onnx_path,
        "--output_dir", f"{openvino_path}/fp16",
        "--log_level", "ERROR"
    ])


def quantization(model_meta: ModelMeta) -> None:
    logging.info(f"{model_meta.name} Model Quantization...")

    frames = []
    video_path = "outputs/video.mp4"
    for frame in read_all_frames(video_path):
        frame = preprocess(frame, model_meta)[0]
        frames.append(frame)

    frames = torch.tensor(numpy.array(frames))
    dataloader = DataLoader(TensorDataset(frames), batch_size=1)
    dataset = nncf.Dataset(dataloader, lambda item: item[0].numpy())

    model_fp32_xml = f"outputs/model/{model_meta.name}/openvino/fp32/model.xml"
    model_int8_xml = f"outputs/model/{model_meta.name}/openvino/int8/model.xml"
    model_fp32 = ov.Core().read_model(model_fp32_xml)
    model_int8 = nncf.quantize(model_fp32, dataset)
    ov.serialize(model_int8, model_int8_xml)


def main():
    download_video()
    for model_meta in MODEL_MAP.values():
        model = download_model(model_meta)
        convert_torch_to_openvino(model_meta, model)
        quantization(model_meta)


if __name__ == '__main__':
    main()
