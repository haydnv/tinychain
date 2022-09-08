import os
import tinychain as tc
import unittest

from ..process import start_host

TC_HOST = tc.URI(os.getenv("TC_HOST", "http://demo.tinychain.net"))


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TC_HOST.host().startswith("127.0.0.1"):
            cls.host = start_host(f"test_client", [], host_uri=TC_HOST, log_level="debug")
        else:
            cls.host = tc.host.Host(str(TC_HOST))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls.host, "stop"):
            cls.host.stop()
