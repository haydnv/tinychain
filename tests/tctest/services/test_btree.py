import tinychain as tc
import unittest

from num2words import num2words
from .base import PersistenceTest
from ..process import DEFAULT_PORT

LEAD = f"http://127.0.0.1:{DEFAULT_PORT}"
NS = tc.URI("/test")
NAME = "btree"
SCHEMA = tc.btree.Schema((tc.Column("number", tc.Int), tc.Column("word", tc.String, 100)))


class BTreeChainTests(PersistenceTest, unittest.TestCase):
    NAME = "btree"
    NUM_HOSTS = 3

    def service(self, chain_type):
        class Persistent(tc.service.Service):
            VERSION = tc.Version("0.0.0")

            __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

            def __init__(self):
                self.tree = chain_type(tc.btree.BTree(SCHEMA))
                tc.service.Service.__init__(self)

        return Persistent()

    def execute(self, hosts):
        row1 = [1, "one"]
        row2 = [2, "two"]

        endpoint = (tc.URI(tc.service.Service) + NS).extend(NAME, "0.0.0", "tree")

        hosts[0].put(endpoint, None, row1)

        for host in hosts:
            actual = host.get(endpoint, (1,))
            self.assertEqual(actual, expected([row1]))

        hosts[1].stop()
        hosts[2].put(endpoint, None, row2)
        hosts[1].start()

        for host in hosts:
            actual = host.get(endpoint, (1,))
            self.assertEqual(actual, expected([row1]))

            actual = host.get(endpoint, (2,))
            self.assertEqual(actual, expected([row2]))

        hosts[2].stop()
        hosts[1].delete(endpoint, (1,))
        hosts[2].start()

        for host in hosts:
            actual = host.get(endpoint)
            self.assertEqual(actual, expected([row2]))

        n = 100
        for i in range(n):
            hosts[0].put(endpoint, None, (i, num2words(i)))

        for host in hosts:
            self.assertEqual(host.get(endpoint.append("count")), n)

def expected(rows):
    return {str(tc.URI(tc.btree.BTree)): [tc.to_json(SCHEMA), rows]}


def printlines(n):
    for _ in range(n):
        print()
