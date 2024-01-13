from exp import dynamic_shape, ov_preprocess, simple_preprocess
from exp.exp_util import parse_exp_args


def test_dynamic_shape():
    cmd = ["-m", "resnet50", "-mt", "int8"]
    args = parse_exp_args(cmd)
    dynamic_shape.main(args)


def test_ov_preprocess():
    cmd = ["-m", "resnet50", "-mt", "int8"]
    args = parse_exp_args(cmd)
    ov_preprocess.main(args)


def test_simple_preprocess():
    cmd = ["-m", "resnet50", "-mt", "int8"]
    args = parse_exp_args(cmd)
    simple_preprocess.main(args)
