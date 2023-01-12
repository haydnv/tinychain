import unittest

import rjwt
import tinychain as tc

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
        actor = rjwt.Actor('/')

        hosts = []

        hosts.append(start_host(NS, http_port=8702, public_key=actor.public_key, replicate=LEAD))

        hosts.append(start_host(NS, http_port=8703, public_key=actor.public_key, replicate=LEAD))

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/replicas"))
            print()

        hosts[0].put("/lib", "test_library", tc.URI(LEAD, "lib", "test_library"))

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/test_library/replicas"))
            print()

        print()
        hosts.append(start_host(NS, http_port=8704, public_key=actor.public_key, replicate=LEAD))
        print()

        for i in range(len(hosts)):
            print()
            print(f"host {i} replicas", hosts[i].get("/lib/test_library/replicas"))
            print()

        hosts[0].install(TestLibV0())
        print()

        for host in hosts:
            print(host)
            self.assertEqual(host.get(tc.URI(TestLibV0).path() + "hello"), "Hello, World!")

        print()

        hosts.append(start_host(NS, [], http_port=8705, public_key=actor.public_key, replicate=LEAD, wait_time=2))

        endpoint = tc.URI(TestLibV0).path() + "hello"

        for i in range(len(hosts)):
            self.assertEqual(hosts[i].get(endpoint), "Hello, World!")

        hosts[2].stop()

        print()
        print("host stopped")
        print()

        hosts[1].update(TestLibV1())

        hosts[2].start(wait_time=2)

        print()
        print("host started")
        print()

        for host in hosts:
            self.assertEqual(host.get(endpoint), "Hello, World!")

        hosts.append(start_host(NS, http_port=8706, public_key=actor.public_key, replicate=LEAD))

        for host in hosts:
            self.assertEqual(host.get(tc.URI(TestLibV1).path() + "hello", "Again"), "Hello, Again!")


def printlines(n):
    for _ in range(n):
        print()
