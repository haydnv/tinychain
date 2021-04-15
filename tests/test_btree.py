import testutils
import tinychain as tc
import unittest


ENDPOINT = "/transact/hypothetical"


class BTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_btree")

    def testCreate(self):
        schema = tc.BTree.Schema(tc.Column("word", tc.String, 100), tc.Column("number", tc.Int))

        cxt = tc.Context()
        cxt.tree = tc.BTree(schema)
        print(tc.to_json(cxt))

        self.host.post(ENDPOINT, cxt)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()

