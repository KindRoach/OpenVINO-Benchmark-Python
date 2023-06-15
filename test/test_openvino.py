import unittest

from openvino_benchmark import sync_infer, async_infer, multi_stream_infer


class TestingSyncInfer(unittest.TestCase):
    def test_sync_decode(self):
        cmd = "-t 5"
        test_args = sync_infer.parse_args(cmd.split())
        sync_infer.main(test_args)

    def test_main_fp32(self):
        cmd = "-t 5 -p fp32"
        test_args = sync_infer.parse_args(cmd.split())
        sync_infer.main(test_args)

    def test_main_fp16(self):
        cmd = "-t 5 -p fp16"
        test_args = sync_infer.parse_args(cmd.split())
        sync_infer.main(test_args)

    def test_main_inference_only(self):
        cmd = "-t 5 -io"
        test_args = sync_infer.parse_args(cmd.split())
        sync_infer.main(test_args)


class TestingAsyncInfer(unittest.TestCase):
    def test_main(self):
        cmd = "-t 5"
        test_args = async_infer.parse_args(cmd.split())
        async_infer.main(test_args)


class TestingMultiStreamInfer(unittest.TestCase):
    def test_main(self):
        cmd = "-t 5 -n 2"
        test_args = multi_stream_infer.parse_args(cmd.split())
        multi_stream_infer.main(test_args)
