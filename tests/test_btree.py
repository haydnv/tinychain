import testutils
import tinychain as tc
import unittest


ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.BTree.Schema(tc.Column("word", tc.String, 100), tc.Column("number", tc.Int))


class BTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_btree")

    def testCreate(self):
        cxt = tc.Context()
        cxt.tree = tc.BTree(SCHEMA)
        cxt.result = tc.After(cxt.tree.insert(("one", 1)), cxt.tree.count())

        count = self.host.post(ENDPOINT, cxt)
        self.assertEqual(count, 1)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()

