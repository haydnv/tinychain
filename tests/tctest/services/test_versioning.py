import unittest
import tinychain as tc

from ..process import start_host


class TestLibV0(tc.app.Library):
    __uri__ = tc.URI("/lib/test")

    # TODO: move into tc.app.Library
    def __json__(self):
        return tc.app.Library.__json__(self)[str(self.__uri__)]

    @tc.post
    def hello(self):
        return "Hello, World!"


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_versioning", [])

    def testCreateLib(self):
        self.host.put("/lib", "test", {})
        self.host.put("/lib/test", "libhello", TestLibV0())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
