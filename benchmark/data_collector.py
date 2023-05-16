import argparse
from pathlib import Path


def main(args) -> None:
    log_dir = Path(args.log_dir)
    total_fps = dict()
    for dir in log_dir.iterdir():
        n_process = int(dir.name)
        total_fps[n_process] = []
        for file in dir.iterdir():
            with file.open("r", encoding="utf-8") as f:
                fps = float(f.readline().strip().split()[1])
                total_fps[n_process].append(fps)
        total_fps[n_process] = sum(total_fps[n_process])

    for n, fps in sorted(total_fps.items()):
        print(f"total fps for {n:3} processes is {fps:8.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str)
    args = parser.parse_args()

    main(args)
