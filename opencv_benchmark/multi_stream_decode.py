import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from tqdm import tqdm

from utils import read_frames


def main(args) -> None:
    video_path = "outputs/video.mp4"
    with tqdm(unit="frame") as pbar:
        def decode_stream(idx: int):
            for frame in read_frames(video_path, args.run_time):
                pbar.update(1)

        with ThreadPoolExecutor(args.n_stream) as pool:
            for i in range(args.n_stream):
                pool.submit(decode_stream, i)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_stream", type=int, default=os.cpu_count())
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
