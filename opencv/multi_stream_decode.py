from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from utils import read_frames, IMG_SIZE

N_STREAM = 8
N_FRAMES = 10000

video_path = "outputs/video.mp4"
with tqdm(total=N_FRAMES * N_STREAM) as pbar:
    def decode_stream(idx: int):
        for i, frame in enumerate(read_frames(video_path, N_FRAMES)):
            inputs = cv2.resize(src=frame, dsize=(IMG_SIZE, IMG_SIZE))
            inputs = np.expand_dims(inputs.transpose(2, 0, 1), 0)
            pbar.update(1)


    with ThreadPoolExecutor(N_STREAM) as pool:
        for i in range(N_STREAM):
            pool.submit(decode_stream, i)
