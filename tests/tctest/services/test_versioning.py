import unittest
import tinychain as tc

from ..process import start_host


class TestLibV0(tc.app.Library):
    __uri__ = tc.URI("/lib/test/libhello")

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
        self.hosts[0].put("/lib/test", "libhello", TestLibV0())
        self.assertEqual(self.hosts[0].get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        self.hosts[0].stop()

        print()
        print("host stopped")
        print()

        self.hosts[0].start()

        print()
        print("host started")
        print()

        self.assertEqual(self.hosts[0].get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

    @classmethod
    def tearDownClass(cls) -> None:
        for host in cls.hosts:
            host.stop()
