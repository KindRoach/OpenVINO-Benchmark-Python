import subprocess

import torch
from torchvision.models import resnet50, ResNet50_Weights

# download model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# convert to onnx
onnx_path = "outputs/model.onnx"
dummy_input = torch.randn(1, 3, 224, 224)
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

# pot quantization
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline

dmz_inputs = torch.randint(0, 256, [300, 3, 224, 224], dtype=torch.float32)


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
compressed_model_paths = save_model(
    model=compressed_model,
    save_path=f"{openvino_path}/int8",
    model_name="model",
)
