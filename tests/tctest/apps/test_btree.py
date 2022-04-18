import tinychain as tc
import unittest

from num2words import num2words
from .base import PersistenceTest
from ..process import DEFAULT_PORT

ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.btree.Schema((tc.Column("number", tc.Int), tc.Column("word", tc.String, 100)))


class BTreeChainTests(PersistenceTest, unittest.TestCase):
    NAME = "btree"

    def app(self, chain_type):
        class Persistent(tc.app.App):
            __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/test/btree")

            def __init__(self):
                self.tree = chain_type(tc.btree.BTree(SCHEMA))
                tc.app.App.__init__(self)

        return Persistent()

    def execute(self, hosts):
        row1 = [1, "one"]
        row2 = [2, "two"]

        hosts[0].put("/test/btree/tree", None, row1)
        for host in hosts:
            actual = host.get("/test/btree/tree", (1,))
            self.assertEqual(actual, expected([row1]))

        hosts[1].stop()
        hosts[2].put("/test/btree/tree", None, row2)
        hosts[1].start()

        for host in hosts:
            actual = host.get("/test/btree/tree", (1,))
            self.assertEqual(actual, expected([row1]))

            actual = host.get("/test/btree/tree", (2,))
            self.assertEqual(actual, expected([row2]))

        hosts[2].stop()
        hosts[1].delete("/test/btree/tree", (1,))
        hosts[2].start()

        for host in hosts:
            actual = host.get("/test/btree/tree")
            self.assertEqual(actual, expected([row2]))

        n = 100
        for i in range(n):
            hosts[0].put("/test/btree/tree", None, (i, num2words(i)))

        for host in hosts:
            self.assertEqual(host.get("/test/btree/tree/count"), n)


def expected(rows):
    return {str(tc.uri(tc.btree.BTree)): [tc.to_json(SCHEMA), rows]}
