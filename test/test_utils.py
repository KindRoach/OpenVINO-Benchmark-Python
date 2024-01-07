from typing import Tuple

import cv2
import numpy
import timm
import torch
from PIL import Image
from numpy.testing import assert_array_equal
from openvino.runtime import Core
from torchvision.transforms import transforms

from run_infer import load_model
from utils import TEST_IMAGE_PATH, preprocess


def torch_predict(model_name: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # load torch model
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    cfg = model.pretrained_cfg

    # torchvision preprocess
    input_image = Image.open(TEST_IMAGE_PATH)
    preprocess = transforms.Compose([
        transforms.Resize(cfg["input_size"][-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg["mean"],
            std=cfg["std"]
        ),
    ])

    # torch predict
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        torch_output = model(input_batch)

    torch_output = torch_output.numpy()
    return torch_output.max(axis=1), torch_output.argmax(axis=1)


def ov_predict(model_name: str, model_type: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # load torch model
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.pretrained_cfg

    frame = cv2.imread(TEST_IMAGE_PATH)
    frame = preprocess(frame, cfg["input_size"], cfg["mean"], cfg["std"])
    model = load_model(Core(), model_name, model_type, "CPU")
    infer_req = model.create_infer_request()
    infer_req.infer(frame)
    ov_output = infer_req.get_output_tensor().data
    return ov_output.max(axis=1), ov_output.argmax(axis=1)


def test_ov_quantization():
    model = "resnet50"
    torch_confidence, torch_label = torch_predict(model)
    ov_confidence, ov_label = ov_predict(model, "int8")
    assert_array_equal(torch_label, ov_label)
