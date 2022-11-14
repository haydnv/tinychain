import unittest
import tinychain as tc

from ..process import start_host


class TestLibV0(tc.app.Library):
    __uri__ = tc.URI("/lib/test/libhello")

    # TODO: remove this override and use the URI key to specify the replication master
    def __json__(self):
        return tc.app.Library.__json__(self)[str(self.__uri__)]

    @tc.get
    def hello(self):
        return "Hello, World!"


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_versioning", [])

    def testCreateLib(self):
        self.host.put("/lib/test")
        self.host.put("/lib/test/libhello", "0.0.1", TestLibV0())
        self.assertEqual(self.host.get("/lib/test/libhello/0.0.1/hello"), "Hello, World!")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
