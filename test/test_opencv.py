import unittest
from unittest.mock import Mock

from opencv import sync_decode, multi_stream_decode


class TestingSyncDecode(unittest.TestCase):
    def test_sync_decode(self):
        mock_args = Mock()
        mock_args.run_time = 5
        sync_decode.main(mock_args)


class TestingMultiStreamDecode(unittest.TestCase):
    def test_main(self):
        mock_args = Mock()
        mock_args.run_time = 5
        mock_args.n_stream = 2
        multi_stream_decode.main(mock_args)
