import threading

import cv2
import numpy as np
from tqdm import tqdm

N_FRAME = 10000
IMG_SIZE = 224


def total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def read_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    i = 0
    while i < N_FRAME:
        success, frame = cap.read()
        if success:
            i += 1
            yield frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()


n_stream = 8
video_path = "outputs/video.mp4"
frame_count = total_frames(video_path)
threads = []
with tqdm(total=N_FRAME * n_stream) as pbar:
    def infer_stream(idx: int):
        for i, frame in enumerate(read_frames(video_path)):
            inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            pbar.update(1)


    for i in range(n_stream):
        t = threading.Thread(target=infer_stream, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
