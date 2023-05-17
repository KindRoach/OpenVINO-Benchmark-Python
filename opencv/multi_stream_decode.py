import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from utils import read_frames

N_FRAMES = 10000


def main(args) -> None:
    video_path = "outputs/video.mp4"
    with tqdm(total=N_FRAMES * args.n_stream) as pbar:
        def decode_stream(idx: int):
            for frame in read_frames(video_path, N_FRAMES):
                pbar.update(1)

        with ThreadPoolExecutor(args.n_stream) as pool:
            for i in range(args.n_stream):
                pool.submit(decode_stream, i)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args)
