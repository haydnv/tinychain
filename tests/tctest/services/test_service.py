import unittest

import rjwt
import tinychain as tc

from ..process import start_host

LEAD = "http://127.0.0.1:8702"
NS = tc.URI("/test_service")


class TestService(tc.service.Service):
    NAME = "hello"
    VERSION = tc.Version("0.0.0")

    __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

    @tc.get
    def hello(self, name: tc.String) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)


class ServiceVersionTests(unittest.TestCase):
    def testCreateService(self):
        actor = rjwt.Actor("/")

        hosts = [
            start_host(NS, http_port=8702, public_key=actor.public_key, replicate=LEAD),
            start_host(NS, http_port=8703, public_key=actor.public_key, replicate=LEAD),
        ]

        hosts[0].create_namespace(actor, tc.URI(tc.service.Service), NS, LEAD)

        print()
        print()

        endpoint = tc.URI(TestService).path()[:-2]

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get(endpoint.append("replicas")))
            print()

        print()

        hosts.append(
            start_host(NS, http_port=8704, public_key=actor.public_key, replicate=LEAD)
        )

        print()

        hosts[0].install(actor, TestService())

        endpoint = tc.URI(TestService).path().append("hello")
        for host in hosts:
            self.assertEqual(host.get(endpoint, "World"), "Hello, World!")

        hosts.append(
            start_host(NS, http_port=8705, public_key=actor.public_key, replicate=LEAD)
        )

        for host in hosts:
            self.assertEqual(host.get(endpoint, "World"), "Hello, World!")

        hosts[2].stop()

        hosts[2].start()

        for host in hosts:
            self.assertEqual(host.get(endpoint, "World"), "Hello, World!")


def printlines(n):
    for _ in range(n):
        print()
