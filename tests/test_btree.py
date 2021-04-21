import random
import testutils
import tinychain as tc
import unittest

from num2words import num2words


ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.BTree.Schema(tc.Column("number", tc.Int), tc.Column("word", tc.String, 100))


class BTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_btree")

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
    __uri__ = tc.URI("/test/btree")

    def _configure(self):
        self.tree = tc.Chain.Block(tc.BTree(SCHEMA))


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hosts = []
        for i in range(1):
            hosts.append(testutils.start_host(f"test_btree_{i}", [Persistent]))

        cls.hosts = hosts

    def testInsert(self):
        row = ["one", 1]

        self.hosts[0].put("/test/btree/tree", None, row)
        actual = self.hosts[0].get("/test/btree/tree", ("one"))
        self.assertEqual(actual, expected([row]))

    @classmethod
    def tearDownClass(cls):
        for host in cls.hosts:
            host.stop()

def expected(rows):
    return {str(tc.uri(tc.BTree)): [tc.to_json(SCHEMA), rows]}


if __name__ == "__main__":
    unittest.main()

