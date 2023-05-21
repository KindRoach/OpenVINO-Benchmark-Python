import argparse

from tqdm import tqdm

from utils import read_frames


def main(args) -> None:
    video_path = "outputs/video.mp4"
    with tqdm(unit="frame") as pbar:
        for frame in read_frames(video_path, args.run_time):
            pbar.update(1)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--run_time", type=int, default=60)
    args = parser.parse_args()

    main(args)
