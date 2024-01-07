from dataclasses import dataclass
from typing import List

from simple_parsing import choice, ArgumentParser

from utils import MODEL_LIST


@dataclass
class ExpArgs:
    model: str = choice(*MODEL_LIST, "all", alias=["-m"], default="resnet50")
    model_type: str = choice("fp32", "fp16", "int8", "all", alias=["-mt"], default="int8")


def parse_exp_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(ExpArgs, dest="arguments")
    return parser.parse_args(args).arguments
