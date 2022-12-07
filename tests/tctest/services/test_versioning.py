import unittest
import tinychain as tc

from ..process import start_host


class TestLibV0(tc.app.Library):
    __uri__ = tc.URI("http://127.0.0.1:8702/lib/test/libhello")

    @tc.get
    def hello(self):
        return "Hello, World!"


class TestLibV1(tc.app.Library):
    __uri__ = tc.URI("/lib/test/libhello")

    @tc.get
    def hello(self, name: tc.String):
        return tc.String("Hello, {{name}}!").render(name=name)


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hosts = [
            start_host("test_versioning", [], http_port=port, replicate="http://127.0.0.1:8702")
            for port in range(8702, 8704)]

    def testCreateLib(self):
        self.hosts[0].put("/lib", "test", tc.URI("http://127.0.0.1:8702/lib/test"))

        for i in range(len(self.hosts)):
            print(f"host {i} replicas", self.hosts[i].get("/lib/test/replicas"))
        return

        self.hosts[2].put("/lib/test", "libhello", TestLibV0())

        for i in range(len(self.hosts)):
            print(f"host {i}")
            self.assertEqual(self.hosts[i].get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        self.hosts[3].stop()

        print()
        print("host stopped")
        print()

        self.hosts[3].start()

        print()
        print("host started")
        print()

        for host in hosts:
            self.assertEqual(host.get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        self.hosts[0].put("/lib/test/libhello", "0.0.1", TestLibV1())

        for host in hosts:
            self.assertEqual(host.get("/lib/test/libhello/0.0.1/hello", "Again"), "Hello, Again!")

    @classmethod
    def tearDownClass(cls) -> None:
        for host in cls.hosts:
            host.stop()
