import unittest

from ..process import start_local_host


class HostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = cls.__name__.lower()
        if name.endswith("test"):
            name = name[:-4]
        elif name.endswith("tests"):
            name = name[:-5]

        cls.host = start_local_host(f"test_{name}")

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()
