import os
import tinychain as tc
import unittest

from ..process import start_host

LEAD = "http://127.0.0.1:8702"
NS = tc.URI("/test_service")


class TestService(tc.service.Service):
    NAME = "echo"
    VERSION = tc.Version("0.0.0")

    __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

    @tc.get
    def hello(self, name: tc.String) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)


class ServiceVersionTests(unittest.TestCase):
    def testCreateService(self):
        key = os.urandom(32)

        hosts = [
            start_host(NS, http_port=8702, symmetric_key=key, wait_time=5),
            start_host(NS, http_port=8703, symmetric_key=key, wait_time=10),
        ]

        printlines(5)

        endpoint = tc.URI(tc.service.Service)
        print(endpoint)

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get(endpoint.append("replicas")))
            print()

        print()

        hosts.append(start_host(NS, http_port=8704, symmetric_key=key, wait_time=10))

        print()

        hosts[0].install(TestService())

        endpoint = tc.URI(TestService).path().append("hello")
        for host in hosts:
            self.assertEqual(host.get(endpoint, "World"), "Hello, World!")

        hosts.append(start_host(NS, http_port=8705, symmetric_key=key, wait_time=20))

        for host in hosts:
            actual = host.get(endpoint, "World")
            print(f"{host}: {actual}")
            self.assertEqual(actual, "Hello, World!")

        hosts[2].stop()

        hosts[2].start(wait_time=10)

        for host in hosts:
            self.assertEqual(host.get(endpoint, "World"), "Hello, World!")


def printlines(n):
    for _ in range(n):
        print()
