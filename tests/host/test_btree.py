import random
import tinychain as tc
import unittest

from num2words import num2words
from testutils import PORT, printlines, start_host, PersistenceTest

ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.btree.Schema((tc.Column("number", tc.Int), tc.Column("word", tc.String, 100)))


class BTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_btree")

    def testCreate(self):
        cxt = tc.Context()
        cxt.tree = tc.btree.BTree(SCHEMA)
        cxt.result = tc.After(cxt.tree.insert((1, "one")), cxt.tree.count())

        count = self.host.post(ENDPOINT, cxt)
        self.assertEqual(count, 1)

    def testInsert(self):
        for x in range(0, 100, 10):
            keys = list(range(x))
            random.shuffle(keys)

            cxt = tc.Context()
            cxt.tree = tc.btree.BTree(SCHEMA)
            cxt.inserts = [
                cxt.tree.insert((i, num2words(i)))
                for i in keys]

            cxt.result = tc.After(cxt.inserts, cxt.tree.count())

            result = self.host.post(ENDPOINT, cxt)
            self.assertEqual(result, x)

    def testSlice(self):
        keys = [[i, num2words(i)] for i in range(50)]

        cxt = tc.Context()
        cxt.tree = tc.btree.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree[(1,)])

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected([keys[1]]))

    def testReverse(self):
        keys = [[i, num2words(i)] for i in range(50)]

        cxt = tc.Context()
        cxt.tree = tc.btree.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree.reverse())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(list(reversed(keys))))

    def testSliceRange(self):
        keys = [[i, num2words(i)] for i in range(50)]

        cxt = tc.Context()
        cxt.tree = tc.btree.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree[29:32])

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(keys[29:32]))

    def testDeleteAll(self):
        keys = [(i, num2words(i)) for i in range(100)]

        cxt = tc.Context()
        cxt.tree = tc.btree.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.delete = tc.After(cxt.inserts, cxt.tree.delete())
        cxt.result = tc.After(cxt.delete, cxt.tree)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected([]))

    def testDeleteSlice(self):
        keys = [[i, num2words(i)] for i in range(10)]
        ordered = keys[:25] + keys[35:]

        random.shuffle(keys)

        cxt = tc.Context()
        cxt.tree = tc.btree.BTree.load(SCHEMA, ordered)
        cxt.delete = cxt.tree.delete([slice(25, 35)])
        cxt.result = tc.After(cxt.delete, cxt.tree)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expected(ordered))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class ChainTests(PersistenceTest, unittest.TestCase):
    NAME = "btree"

    def cluster(self, chain_type):
        class Persistent(tc.Cluster, metaclass=tc.Meta):
            __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/test/btree")

            def _configure(self):
                self.tree = chain_type(tc.btree.BTree(SCHEMA))

        return Persistent

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


if __name__ == "__main__":
    unittest.main()

