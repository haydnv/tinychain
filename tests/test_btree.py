import random
import tinychain as tc
import unittest

from num2words import num2words
from testutils import start_host, PORT


ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.BTree.Schema(tc.Column("number", tc.Int), tc.Column("word", tc.String, 100))


class BTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_btree")

    def testCreate(self):
        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.result = tc.After(cxt.tree.insert((1, "one")), cxt.tree.count())

        count = self.host.post(ENDPOINT, cxt)
        self.assertEqual(count, 1)

    def testInsert(self):
        for x in range(0, 100, 10):
            keys = list(range(x))
            random.shuffle(keys)

            cxt = tc.Context()
            cxt.tree = tc.BTree(SCHEMA)
            cxt.inserts = [
                cxt.tree.insert((i, num2words(i)))
                for i in keys]

            cxt.result = tc.After(cxt.inserts, cxt.tree.count())

            result = self.host.post(ENDPOINT, cxt)
            self.assertEqual(result, x)

    def testSlice(self):
        keys = [[i, num2words(i)] for i in range(50)]

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree[(1,)])

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected([keys[1]]))

    def testReverse(self):
        keys = [[i, num2words(i)] for i in range(50)]

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree.reverse())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(list(reversed(keys))))

    def testSliceRange(self):
        keys = [[i, num2words(i)] for i in range(50)]

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree[29:32])

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(keys[29:32]))

    def testDelete(self):
        keys = [(i, num2words(i)) for i in range(100)]

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.delete = tc.After(cxt.inserts, cxt.tree.delete())
        cxt.result = tc.After(cxt.delete, cxt.tree)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected([]))

    def testDeleteSlice(self):
        keys = [[i, num2words(i)] for i in range(100)]
        ordered = expected(keys[:25] + keys[35:])

        random.shuffle(keys)

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.slice = cxt.tree[25:35]
        cxt.delete = tc.After(cxt.inserts, cxt.slice.delete())
        cxt.result = tc.After(cxt.delete, cxt.tree)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, ordered)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class Persistent(tc.Cluster):
    __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/test/btree")

    def _configure(self):
        self.tree = tc.Chain.Block(tc.BTree(SCHEMA))


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hosts = []
        for i in range(3):
            port = PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.uri(Persistent).path()
            host = start_host(f"test_replication_{i}", [Persistent], host_uri=tc.URI(host_uri))
            hosts.append(host)

        cls.hosts = hosts

    def testInsert(self):
        row1 = [1, "one"]
        row2 = [2, "two"]

        self.hosts[0].put("/test/btree/tree", None, row1)
        for host in self.hosts:
            actual = host.get("/test/btree/tree", (1,))
            self.assertEqual(actual, expected([row1]))

        self.hosts[1].stop()
        self.hosts[2].put("/test/btree/tree", None, row2)
        self.hosts[1].start()

        for host in self.hosts:
            actual = host.get("/test/btree/tree", (1,))
            self.assertEqual(actual, expected([row1]))

            actual = host.get("/test/btree/tree", (2,))
            self.assertEqual(actual, expected([row2]))

        self.hosts[2].stop()
        self.hosts[1].delete("/test/btree/tree", (1,))
        self.hosts[2].start()

        for host in self.hosts:
            actual = host.get("/test/btree/tree")
            self.assertEqual(actual, expected([row2]))

        n = 100
        for i in range(n):
            self.hosts[0].put("/test/btree/tree", None, (i, num2words(i)))

        self.assertEqual(self.hosts[1].get("/test/btree/tree/count"), n)

    @classmethod
    def tearDownClass(cls):
        for host in cls.hosts:
            host.stop()


def expected(rows):
    return {str(tc.uri(tc.BTree)): [tc.to_json(SCHEMA), rows]}


if __name__ == "__main__":
    unittest.main()

