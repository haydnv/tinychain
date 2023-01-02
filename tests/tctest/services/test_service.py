import unittest
import tinychain as tc

from ..process import start_host

NAME = "test_service"
LEAD = "http://127.0.0.1:8702"
DIR = tc.URI(LEAD + "/service/test")


class TestServiceV0(tc.app.Service):
    HOST = tc.URI(LEAD)
    NS = tc.URI("/test")
    NAME = "hello"
    VERSION = tc.Version("0.0.0")

    __uri__ = HOST + tc.URI(tc.app.Service) + NS.append(NAME) + VERSION

    @tc.get
    def hello(self) -> tc.String:
        return "Hello, World!"


class ServiceVersionTests(unittest.TestCase):
    def testCreateService(self):
        hosts = [
            start_host(NAME, [], http_port=8702, replicate=LEAD),
            start_host(NAME, [], http_port=8703, replicate=LEAD),
        ]

        hosts[0].put("/service", "test", DIR)

        print()
        print()

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/service/test/replicas"))
            print()

        print()
        hosts.append(start_host(NAME, [], http_port=8704, replicate=LEAD))
        print()

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/service/test/replicas"))
            print()

        hosts[0].install(TestServiceV0())
        print()

        for host in hosts:
            print(host)
            self.assertEqual(hosts[i].get("/service/test/hello/0.0.0/hello"), "Hello, World!")

        hosts.append(start_host(NAME, [], http_port=8705, replicate=LEAD))

        for host in hosts:
            print(host)
            self.assertEqual(hosts[i].get("/service/test/hello/0.0.0/hello"), "Hello, World!")

        hosts[2].stop()

        print()
        print("host stopped")
        print()

        hosts[2].start()

        print()
        print("host started")
        print()

        for host in hosts:
            self.assertEqual(host.get("/service/test/hello/0.0.0/hello"), "Hello, World!")


def printlines(n):
    for _ in range(n):
        print()
