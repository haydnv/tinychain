import unittest

from ..process import start_local_host


class HostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls, request_ttl=3):
        name = cls.__name__.lower()
        if name.endswith("test"):
            name = name[:-4]
        elif name.endswith("tests"):
            name = name[:-5]

        cls.host = start_local_host(f"/test_{name}", request_ttl=request_ttl)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()
