import unittest
import tinychain as tc

from ..process import start_host

NAME = "test_library"
LEAD = "http://127.0.0.1:8702"
DIR = tc.URI(LEAD + "/lib/test")


class TestLibV0(tc.app.Library):
    HOST = tc.URI(LEAD)
    NS = tc.URI("/test")
    NAME = "libhello"
    VERSION = tc.Version("0.0.0")

    __uri__ = HOST + tc.URI(tc.app.Library) + NS.append(NAME) + VERSION

    @tc.get
    def hello(self) -> tc.String:
        return "Hello, World!"


class TestLibV1(tc.app.Library):
    URI = DIR + "libhello"
    VERSION = tc.Version("0.0.1")

    __uri__ = URI + VERSION

    @tc.get
    def hello(self, name: tc.String) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)


class LibraryVersionTests(unittest.TestCase):
    def testCreateLib(self):
        hosts = []

        hosts.append(start_host(NAME, [], http_port=8702, replicate=LEAD))

        hosts.append(start_host(NAME, [], http_port=8703, replicate=LEAD))

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/replicas"))
            print()

        hosts[0].put("/lib", "test", DIR)

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/test/replicas"))
            print()

        print()
        hosts.append(start_host(NAME, [], http_port=8704, replicate=LEAD))
        print()

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/test/replicas"))
            print()

        hosts[0].install(TestLibV0())
        print()

        hosts.append(start_host(NAME, [], http_port=8705, replicate=LEAD))

        for i in range(len(hosts)):
            self.assertEqual(hosts[i].get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        hosts[2].stop()

        print()
        print("host stopped")
        print()

        hosts[1].update(TestLibV1())

        hosts[2].start()

        print()
        print("host started")
        print()

        for host in hosts:
            self.assertEqual(host.get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        hosts.append(start_host(NAME, [], http_port=8706, replicate=LEAD))

        for host in hosts:
            self.assertEqual(host.get("/lib/test/libhello/0.0.1/hello", "Again"), "Hello, Again!")


def printlines(n):
    for _ in range(n):
        print()
