import unittest

from opencv_benchmark import sync_decode, multi_stream_decode


class TestingSyncDecode(unittest.TestCase):
    def test_sync_decode(self):
        cmd = "-t 5"
        test_args = sync_decode.parse_args(cmd.split())
        sync_decode.main(test_args)


class TestingMultiStreamDecode(unittest.TestCase):
    def test_main(self):
        cmd = "-t 5 -n 2"
        test_args = multi_stream_decode.parse_args(cmd.split())
        multi_stream_decode.main(test_args)
