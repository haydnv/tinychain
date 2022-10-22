import unittest
import tinychain as tc

from ..process import start_host


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_versioning", [])

    def testCreateDir(self):
        self.host.put("/lib", "test", {})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
