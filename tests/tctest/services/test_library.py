import os
import tinychain as tc
import unittest

from ..process import start_host

LEAD = "http://127.0.0.1:8702"
NS = tc.URI("/test_library")
NAME = "libhello"


class TestLibV0(tc.service.Library):
    VERSION = tc.Version("0.0.0")

    __uri__ = tc.service.library_uri(LEAD, NS, NAME, VERSION)

    @tc.get
    def hello(self) -> tc.String:
        return "Hello, World!"


class TestLibV1(tc.service.Library):
    VERSION = tc.Version("0.0.1")

    __uri__ = tc.service.library_uri(LEAD, NS, NAME, VERSION)

    @tc.get
    def hello(self, name: tc.String) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)


class LibraryVersionTests(unittest.TestCase):
    def testCreateLib(self):
        key = os.urandom(32)
        hosts = []

        hosts.append(start_host(NS, http_port=8702, symmetric_key=key))
        hosts.append(start_host(NS, http_port=8703, symmetric_key=key))

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/replicas"))
            print()

        hosts.append(start_host(NS, http_port=8704, symmetric_key=key))

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/replicas"))
            print()

        print()
        print("INSTALL LIBRARY")
        hosts[1].install(TestLibV0())
        print()

        for i in range(len(hosts)):
            print()
            print(f"host {i} library replicas", hosts[i].get("/lib/test_library/replicas"))
            print()

        for host in hosts:
            print(host)

            self.assertEqual(host.get(tc.URI(TestLibV0).path() + "hello"), "Hello, World!")

        print()

        print("starting additional host...")
        hosts.append(start_host(NS, [], http_port=8705, symmetric_key=key))
        print("started")

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/test_library/replicas"))
            print()

        endpoint = tc.URI(TestLibV0).path() + "hello"

        for i in range(len(hosts)):
            self.assertEqual(hosts[i].get(endpoint), "Hello, World!")

        hosts[1].stop()

        print()
        print(f"{hosts[1]} stopped")
        print()

        hosts[2].update(TestLibV1())

        print()
        print(f"restarting {hosts[1]}...")
        print()

        hosts[1].start(wait_time=10)

        print()
        print("host started")
        print()

        endpoint = tc.URI(TestLibV1).path() + "hello"

        for i in range(len(hosts)):
            self.assertEqual(hosts[i].get(endpoint, "Test"), "Hello, Test!")

        print()
        print("starting additional host...")
        hosts.append(start_host(NS, http_port=8706, symmetric_key=key))
        print("started")
        print()

        for host in hosts:
            self.assertEqual(
                host.get(endpoint, "Again"), "Hello, Again!"
            )


def printlines(n):
    for _ in range(n):
        print()
