import unittest

from opencv_benchmark import sync_decode, multi_stream_decode


class TestingSyncDecode(unittest.TestCase):
    def test_sync_decode(self):
        test_args = sync_decode.parse_args([])
        test_args.run_time = 5
        sync_decode.main(test_args)


class TestingMultiStreamDecode(unittest.TestCase):
    def test_main(self):
        test_args = multi_stream_decode.parse_args([])
        test_args.run_time = 5
        test_args.n_stream = 2
        multi_stream_decode.main(test_args)
