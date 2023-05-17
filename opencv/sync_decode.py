import argparse

from tqdm import tqdm

from utils import read_frames

N_FRAMES = 10000


def main(args) -> None:
    video_path = "outputs/video.mp4"
    with tqdm(total=N_FRAMES) as pbar:
        for frame in read_frames(video_path, N_FRAMES):
            pbar.update(1)

        frames = pbar.format_dict["n"]
        seconds = pbar.format_dict["elapsed"]

    print(f"fps: {frames / seconds:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
