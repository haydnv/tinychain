import unittest
import tinychain as tc

from ..process import start_host

NAME = "test_versioning"
LEAD = "http://127.0.0.1:8702"
DIR = tc.URI(LEAD + "/lib/test")


class TestLibV0(tc.app.Library):
    __uri__ = DIR + "libhello"

    @tc.get
    def hello(self) -> tc.String:
        return "Hello, World!"


class TestLibV1(tc.app.Library):
    __uri__ = DIR + "libhello"

    @tc.get
    def hello(self, name: tc.String) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)


class LibraryTests(unittest.TestCase):
    def testCreateLib(self):
        hosts = [
            start_host(NAME, [], http_port=8702, replicate=LEAD),
            start_host(NAME, [], http_port=8703, replicate=LEAD),
        ]

        hosts[0].put("/lib", "test", DIR)

        print()
        print()

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

        hosts[0].put("/lib/test", "libhello", TestLibV0())
        print()

        hosts.append(start_host(NAME, [], http_port=8705, replicate=LEAD))

        for i in range(len(hosts)):
            self.assertEqual(hosts[i].get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        hosts[2].stop()

        print()
        print("host stopped")
        print()

        hosts[2].start()

        print()
        print("host started")
        print()

        for host in hosts:
            self.assertEqual(host.get("/lib/test/libhello/0.0.0/hello"), "Hello, World!")

        hosts[1].put("/lib/test/libhello", "0.0.1", TestLibV1())

        hosts.append(start_host(NAME, [], http_port=8706, replicate=LEAD))

        for host in hosts:
            self.assertEqual(host.get("/lib/test/libhello/0.0.1/hello", "Again"), "Hello, Again!")
