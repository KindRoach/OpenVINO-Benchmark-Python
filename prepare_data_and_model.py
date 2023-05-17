import logging
import subprocess
import urllib.request
from pathlib import Path

import torch
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline
from openvino.tools.pot import load_model, save_model
from torch.nn import Module

from utils import MODEL_MAP, ModelMeta


def download_video() -> None:
    video_path = "outputs/video.mp4"
    if not Path(video_path).exists():
        logging.info("Downloading Video...")
        video_url = "https://s3.amazonaws.com/senkorasic.com/test-media/video/caminandes-llamigos/caminandes_llamigos_1080p.mp4"
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
        "--output_dir", f"{openvino_path}/fp32"
    ])
    subprocess.run([
        "mo",
        "--compress_to_fp16",
        "--input_model", onnx_path,
        "--output_dir", f"{openvino_path}/fp16"
    ])


def quantization(model_meta: ModelMeta) -> None:
    logging.info("Model Quantization...")

    dmz_inputs = torch.rand([300, *model_meta.input_size], dtype=torch.float32)

    class DmzLoader(DataLoader):
        def __init__(self):
            super().__init__(None)

        def __len__(self):
            return len(dmz_inputs)

        def __getitem__(self, index):
            # annotation is set to None
            return dmz_inputs[index], None

    engine = IEEngine(config={"device": "CPU"}, data_loader=DmzLoader())
    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "ANY",
                "stat_subset_size": 300,
                "stat_batch_size": 1
            },
        }
    ]
    pipeline = create_pipeline(algorithms, engine)

    model = load_model(model_config={
        "model_name": "model",
        "model": f"outputs/model/{model_meta.name}/openvino/fp32/model.xml",
        "weights": f"outputs/model/{model_meta.name}/openvino/fp32/model.bin",
    })
    compressed_model = pipeline.run(model=model)

    compress_model_weights(compressed_model)

    save_model(
        model=compressed_model,
        save_path=f"outputs/model/{model_meta.name}/openvino/int8",
        model_name="model",
    )


def main():
    download_video()
    for model_meta in MODEL_MAP.values():
        model = download_model(model_meta)

        convert_torch_to_openvino(model_meta, model)
        quantization(model_meta)


if __name__ == '__main__':
    main()
