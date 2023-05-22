import unittest

from openvino_benchmark import sync_infer, async_infer, multi_stream_infer


class TestingSyncInfer(unittest.TestCase):
    def test_sync_decode(self):
        test_args = sync_infer.parse_ages([])
        test_args.run_time = 5
        sync_infer.main(test_args)


class TestingAsyncInfer(unittest.TestCase):
    def test_main(self):
        test_args = async_infer.parse_ages([])
        test_args.run_time = 5
        test_args.infer_jobs = 2
        async_infer.main(test_args)


class TestingMultiStreamInfer(unittest.TestCase):
    def test_main(self):
        test_args = multi_stream_infer.parse_ages([])
        test_args.run_time = 5
        test_args.n_stream = 2
        multi_stream_infer.main(test_args)
