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
        x = 100

        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.inserts = [cxt.tree.insert((i, num2words(i))) for i in range(x)]
        cxt.result = tc.After(cxt.inserts, cxt.tree.count())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, x)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()

