import argparse
import sys
from typing import List

from tqdm import tqdm

from utils import read_frames_with_time


def main(args) -> None:
    with tqdm(unit="frame") as pbar:
        for frame in read_frames_with_time(args.run_time):
            pbar.update(1)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--run_time", type=int, default=60)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
