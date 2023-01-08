import os
import tinychain as tc
import unittest

from ..process import start_host

TC_HOST = tc.URI(os.getenv("TC_HOST", "http://demo.tinychain.net"))


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TC_HOST.host().startswith("127.0.0.1"):
            name = cls.__name__.lower()
            if name.endswith("test"):
                name = name[:-4]
            elif name.endswith("tests"):
                name = name[:-5]

            cls.host = start_host(f"/test_{name}", host_uri=TC_HOST, log_level="debug")
        else:
            cls.host = tc.host.Host(str(TC_HOST))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls.host, "stop"):
            cls.host.stop()
