import tinychain as tc
import unittest


class CollectionTests(unittest.TestCase):
    def testBTree(self):
        schema = tc.schema.BTree(tc.Column("word", tc.String, 100), tc.Column("number", tc.Int))
        self.assertEqual(tc.to_json(tc.BTree(schema)), {
            '/state/collection/btree': [[
                ["word", '/state/scalar/value/string', 100],
                ["number", '/state/scalar/value/number/int']
            ]]
        })


if __name__ == "__main__":
    unittest.main()

