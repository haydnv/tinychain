import random
import tinychain as tc
import unittest

from num2words import num2words
from .base import PersistenceTest
from ..process import DEFAULT_PORT, start_host

LEAD = f"http://127.0.0.1:{DEFAULT_PORT}"
NS = tc.URI("/test_table")
SCHEMA = tc.table.Schema(
    [tc.Column("name", tc.String, 512)],
    [tc.Column("views", tc.UInt)]).create_index("views", ["views"])


class TableChainTests(PersistenceTest, unittest.TestCase):
    def service(self, chain_type):
        class Persistent(tc.service.Service):
            NAME = "table"
            VERSION = tc.Version("0.0.0")

            __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

            def __init__(self):
                self.table = chain_type(tc.table.Table(SCHEMA))
                tc.service.Service.__init__(self)

        return Persistent()

    def execute(self, hosts):
        row1 = ["one", 1]
        row2 = ["two", 2]

        endpoint = (tc.URI(tc.service.Service) + NS).extend("table", "0.0.0", "table")
        self.assertIsNone(hosts[1].put(endpoint, ["one"], [1]))

        for host in hosts:
            actual = host.get(endpoint, ["one"])
            self.assertEqual(actual, row1)

        hosts[1].stop()
        hosts[2].put(endpoint, ["two"], [2])
        hosts[1].start()

        for host in hosts:
            actual = host.get(endpoint, ["one"])
            self.assertEqual(actual, row1)

            actual = host.get(endpoint, ["two"])
            self.assertEqual(actual, row2)

        hosts[2].stop()
        self.assertIsNone(hosts[1].delete(endpoint, ["one"]))
        hosts[2].start()

        for i in range(len(hosts)):
            actual = hosts[i].get(endpoint)
            self.assertEqual(actual, expected(SCHEMA, [["two", 2]]), f"host {i}")

        for i in range(len(hosts)):
            count = hosts[i].get(endpoint.append("count"))
            self.assertEqual(1, count, f"host {i}")

        self.assertIsNone(hosts[1].delete(endpoint, ["two"]))

        for i in range(len(hosts)):
            count = hosts[i].get(endpoint.append("count"))
            self.assertEqual(0, count, f"host {i}")

        total = 100
        for n in range(1, total):
            i = random.choice(range(self.NUM_HOSTS))

            self.assertIsNone(hosts[i].put(endpoint, [num2words(n)], [n]))

            for i in range(len(hosts)):
                count = hosts[i].get(endpoint.append("count"))
                self.assertEqual(n, count, f"host {i}")


def expected(schema, rows):
    return {str(tc.URI(tc.table.Table)): [tc.to_json(schema), rows]}
