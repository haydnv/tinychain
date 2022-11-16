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
        cls.hosts = [
            start_host("test_versioning", [], http_port=port, replicate="http://127.0.0.1:8702")
            for port in range(8702, 8703)]

    def testCreateLib(self):
        self.hosts[0].put("/lib", "test", tc.URI("/lib/test"))
        # self.host.put("/lib/test/libhello", "0.0.1", TestLibV0())
        # self.assertEqual(self.host.get("/lib/test/libhello/0.0.1/hello"), "Hello, World!")
        #
        # self.host.stop()
        #
        # print()
        # print("host stopped")
        # print()
        #
        # self.host.start()
        #
        # print()
        # print("host started")
        # print()
        #
        # self.assertEqual(self.host.get("/lib/test/libhello/0.0.1/hello"), "Hello, World!")

    @classmethod
    def tearDownClass(cls) -> None:
        for host in cls.hosts:
            host.stop()
