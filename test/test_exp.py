from exp import dynamic_shape, ov_preprocess, auto_batch
from exp.exp_args import parse_exp_args


def test_dynamic_shape():
    cmd = ["-m", "resnet_50", "-mt", "int8"]
    args = parse_exp_args(cmd)
    dynamic_shape.main(args)


def test_ov_preprocess():
    cmd = ["-m", "resnet_50", "-mt", "int8"]
    args = parse_exp_args(cmd)
    ov_preprocess.main(args)


def test_auto_batch():
    cmd = ["-m", "resnet_50", "-mt", "int8"]
    args = parse_exp_args(cmd)
    auto_batch.main(args)
