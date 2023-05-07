import subprocess
import urllib.request

import torch
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline
from openvino.tools.pot import load_model, save_model
from torch.nn import Module
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

IMG_SIZE = 480


def download_video() -> None:
    video_url = "https://storage.openvinotoolkit.org/data/test_data/videos/car-detection.mp4"
    urllib.request.urlretrieve(video_url, "outputs/video.mp4")


def download_model() -> Module:
    model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    model.eval()
    return model


def convert_torch_to_openvino(model: Module) -> str:
    # convert to onnx
    onnx_path = "outputs/model.onnx"
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, (dummy_input,), onnx_path)

    # convert to openvino
    openvino_path = "outputs/openvino"
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

    return openvino_path


def quantization(openvino_path: str) -> None:
    dmz_inputs = torch.rand([300, 3, IMG_SIZE, IMG_SIZE], dtype=torch.float32)

    class DmzLoader(DataLoader):
        def __init__(self):
            super().__init__(None)

        def __len__(self):
            return len(dmz_inputs)

        def __getitem__(self, index):
            # annotation is set to None
            return dmz_inputs[index], None

    model = load_model(model_config={
        "model_name": "model",
        "model": f"{openvino_path}/fp32/model.xml",
        "weights": f"{openvino_path}/fp32/model.bin",
    })
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
    compressed_model = pipeline.run(model=model)
    compress_model_weights(compressed_model)
    save_model(
        model=compressed_model,
        save_path=f"{openvino_path}/int8",
        model_name="model",
    )


def main():
    download_video()
    model = download_model()
    path = convert_torch_to_openvino(model)
    quantization(path)


if __name__ == '__main__':
    main()
