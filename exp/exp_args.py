from dataclasses import dataclass
from typing import List

from simple_parsing import choice, ArgumentParser

from utils import MODEL_MAP


@dataclass
class ExpArgs:
    model: str = choice(*MODEL_MAP.keys(), "all", alias=["-m"], default="all")
    model_type: str = choice("fp32", "fp16", "int8", "all", alias=["-mt"], default="all")


def parse_exp_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(ExpArgs, dest="arguments")
    return parser.parse_args(args).arguments
