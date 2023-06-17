import unittest

from run_infer import parse_args, main


class TestingSyncInfer(unittest.TestCase):
    def test_sync_decode(self):
        cmd = "-t 5 -rm sync"
        test_args = parse_args(cmd.split())
        main(test_args)

    def test_main_fp32(self):
        cmd = "-t 5 -mt fp32 -rm sync"
        test_args = parse_args(cmd.split())
        main(test_args)

    def test_main_fp16(self):
        cmd = "-t 5 -mt fp16 -rm sync"
        test_args = parse_args(cmd.split())
        main(test_args)

    def test_main_inference_only(self):
        cmd = "-t 5 -io -rm sync"
        test_args = parse_args(cmd.split())
        main(test_args)

    def test_main_ov_preprocess(self):
        cmd = "-t 5 -io -op -rm sync"
        test_args = parse_args(cmd.split())
        main(test_args)


class TestingAsyncInfer(unittest.TestCase):
    def test_main(self):
        cmd = "-t 5 -rm async"
        test_args = parse_args(cmd.split())
        main(test_args)


class TestingMultiStreamInfer(unittest.TestCase):
    def test_main(self):
        cmd = "-t 5 -rm multi -n 2"
        test_args = parse_args(cmd.split())
        main(test_args)
