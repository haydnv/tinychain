import tinychain as tc
import unittest

from num2words import num2words
from .base import PersistenceTest
from ..process import DEFAULT_PORT

SCHEMA = tc.btree.Schema((tc.Column("number", tc.Int), tc.Column("word", tc.String, 100)))


class Persistent(tc.app.Service):
    HOST = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}")
    NS = tc.URI("/test")
    NAME = "btree"
    VERSION = tc.Version("0.0.0")

    __uri__ = HOST + tc.URI(tc.app.Service) + NS.extend(NAME, VERSION)


class BTreeChainTests(PersistenceTest, unittest.TestCase):
    def service(self, chain_type):
        class _Persistent(Persistent):
            def __init__(self):
                self.tree = chain_type(tc.btree.BTree(SCHEMA))
                tc.app.Service.__init__(self)

        return _Persistent()

    def execute(self, hosts):
        endpoint = tc.URI(Persistent).path().append("tree")

        row1 = [1, "one"]
        row2 = [2, "two"]

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
