import cv2
import numpy as np
from openvino.runtime import Core
from tqdm import tqdm


def total_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def read_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    success, frame = cap.read()
    while success:
        yield frame
        success, frame = cap.read()
    cap.release()


def sync_infer(model, video_path):
    for frame in tqdm(read_frames(video_path), total=total_frames(video_path)):
        inputs = cv2.resize(src=frame, dsize=(224, 224))
        inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
        model(inputs)


def main():
    ie = Core()

    devices = ie.available_devices

    for device in devices:
        device_name = ie.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    model_xml = "outputs/openvino/fp32/model.xml"
    compiled_model = ie.compile_model(model_xml, device_name="CPU")
    sync_infer(compiled_model, "outputs/video.mp4")


if __name__ == '__main__':
    main()
