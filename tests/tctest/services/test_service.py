import unittest
import tinychain as tc

from ..process import start_host

LEAD = "http://127.0.0.1:8702"
NS = tc.URI("/test_service")


class TestServiceV0(tc.service.Service):
    NAME = "hello"
    VERSION = tc.Version("0.0.0")

    __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

    @tc.get
    def hello(self) -> tc.String:
        return "Hello, World!"


class ServiceVersionTests(unittest.TestCase):
    def testCreateService(self):
        hosts = [
            start_host(NS, http_port=8702, replicate=LEAD),
            start_host(NS, http_port=8703, replicate=LEAD),
        ]

        hosts[0].put(tc.URI(tc.service.Service), "test_service", tc.URI(LEAD, "service", "test_service"))

        print()
        print()

        endpoint = tc.URI(TestServiceV0).path()[:-2]

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get(endpoint.append("replicas")))
            print()

        print()
        hosts.append(start_host(NS, http_port=8704, replicate=LEAD))
        print()

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get(endpoint.append("replicas")))
            print()

        hosts[0].install(TestServiceV0())
        print()

        endpoint = tc.URI(TestServiceV0).path().append("hello")
        for host in hosts:
            print(host)
            self.assertEqual(hosts[i].get(endpoint), "Hello, World!")

        hosts.append(start_host(NS, http_port=8705, replicate=LEAD))

        for host in hosts:
            print(host)
            self.assertEqual(hosts[i].get(endpoint), "Hello, World!")

        hosts[2].stop()

        print()
        print("host stopped")
        print()

        hosts[2].start()

        print()
        print("host started")
        print()

        for host in hosts:
            self.assertEqual(host.get(endpoint), "Hello, World!")


def printlines(n):
    for _ in range(n):
        print()
