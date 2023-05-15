import threading

import cv2
import numpy as np
from tqdm import tqdm

from utils import read_frames, IMG_SIZE

N_STREAM = 8
N_FRAMES = 10000

video_path = "outputs/video.mp4"
with tqdm(total=N_FRAMES * N_STREAM) as pbar:
    def infer_stream(idx: int):
        for i, frame in enumerate(read_frames(video_path, N_FRAMES)):
            inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            pbar.update(1)


    threads = []
    for i in range(N_STREAM):
        t = threading.Thread(target=infer_stream, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
