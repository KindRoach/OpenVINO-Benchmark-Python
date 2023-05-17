import argparse
import os

import cv2
import numpy as np
import openvino.runtime as ov
from openvino.runtime import Core, CompiledModel
from tqdm import tqdm

from utils import read_frames, MODEL_MAP, ModelMeta

N_FRAME = 1000


def async_infer(model: CompiledModel, model_meta: ModelMeta, video_path: str, n_jobs: int) -> np.array:
    frame_count = N_FRAME
    outputs = [None] * frame_count
    with tqdm(total=frame_count) as pbar:
        def call_back(request, userdata):
            outputs[userdata] = request.get_output_tensor().data
            # pbar.write(f"frame {userdata} done!")
            pbar.update(1)

        infer_queue = ov.AsyncInferQueue(model, n_jobs)
        infer_queue.set_callback(call_back)

        for i, frame in enumerate(read_frames(video_path, N_FRAME)):
            inputs = cv2.resize(src=frame, dsize=model_meta.input_size[-2:])
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            infer_queue.start_async(inputs, i)

        infer_queue.wait_all()

    return outputs


def main(args) -> None:
    ie = Core()
    ie.set_property("CPU", {
        "NUM_STREAMS": args.infer_jobs,
        "PERFORMANCE_HINT": "THROUGHPUT"
    })

    model_meta = MODEL_MAP[args.model]
    model_xml = f"outputs/model/{model_meta.name}/openvino/{args.model_precision}/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name=args.device)
    video_path = "outputs/video.mp4"
    async_infer(compiled_model, model_meta, video_path, args.infer_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        choices=["CPU", "GPU"] + [f"GPU.{i}" for i in range(8)])
    parser.add_argument("-m", "--model", type=str, default="resnet_50", choices=list(MODEL_MAP.keys()))
    parser.add_argument("-p", "--model_precision", type=str, default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("-n", "--infer_jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args)
