import time

import numpy
import torch
import tqdm
from torchvision.transforms import transforms
from tqdm import tqdm

from utils import preprocess


def run_trochvison_preprocess(seconds: int, batch_size: int, resize_shape: (int, int)):
    shape = (batch_size, 1080, 1920, 3)
    random_frames = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_shape, antialias=True),
        transforms.Normalize(
            mean=(0.1, 0.1, 0.1),
            std=(0.1, 0.1, 0.1)
        ),
    ])

    with tqdm(desc="preprocess by torchvision", unit="frame", unit_scale=batch_size) as pbar:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < seconds:
            processed = torch.concatenate([transformer(frame) for frame in random_frames])
            pbar.update(1)


def run_opencv_preprocess(seconds: int, batch_size: int, resize_shape: (int, int)):
    shape = (batch_size, 1080, 1920, 3)
    random_frames = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    with tqdm(desc="preprocess by opencv and numpy", unit="frame", unit_scale=batch_size) as pbar:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < seconds:
            preprocess(random_frames, (3, resize_shape[0], resize_shape[1]), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
            pbar.update(1)


def main():
    batch_size = 64
    run_opencv_preprocess(10, batch_size, (224, 224))
    run_trochvison_preprocess(10, batch_size, (224, 224))


if __name__ == '__main__':
    main()
