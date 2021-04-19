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
            cxt = tc.Context()
            cxt.tree = tc.BTree(SCHEMA)
            cxt.inserts = [cxt.tree.insert((i, num2words(i))) for i in range(x)]
            cxt.result = tc.After(cxt.inserts, cxt.tree.count())

            result = self.host.post(ENDPOINT, cxt)
            self.assertEqual(result, x)

    def testSlice(self):
        keys = [(i, num2words(i)) for i in range(50)]
        expected = {str(tc.uri(tc.BTree)): [tc.to_json(SCHEMA), [keys[1]]]}

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree[(1,)])

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, tc.to_json(expected))

    def testReverse(self):
        keys = [(i, num2words(i)) for i in range(50)]
        expected = {str(tc.uri(tc.BTree)): [tc.to_json(SCHEMA), list(reversed(keys))]}

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree.reverse())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, tc.to_json(expected))

    def testSliceRange(self):
        keys = [(i, num2words(i)) for i in range(50)]
        expected = {str(tc.uri(tc.BTree)): [tc.to_json(SCHEMA), keys[29:32]]}

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.result = tc.After(cxt.inserts, cxt.tree[29:32])

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, tc.to_json(expected))

    def testDelete(self):
        keys = [(i, num2words(i)) for i in range(29)]
        expected = {str(tc.uri(tc.BTree)): [tc.to_json(SCHEMA), []]}

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert(key) for key in keys]
        cxt.delete = tc.After(cxt.inserts, cxt.tree.delete())
        cxt.result = tc.After(cxt.delete, cxt.tree)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, tc.to_json(expected))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()

