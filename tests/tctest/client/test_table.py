import tinychain as tc
import unittest

from num2words import num2words
from templates import ClientTest

ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.table.Schema(
    [tc.Column("name", tc.String, 512)], [tc.Column("views", tc.UInt)]).create_index("views", ["views"])


class TableTests(ClientTest):
    def testAggregate(self):
        count = 10
        values = [(v % 2,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table.load(SCHEMA, [k + v for k, v in zip(keys, values)])
        cxt.result = cxt.table.aggregate(["views"], lambda group: tc.Tuple(group.count()))

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [[[0], 5], [[1], 5]])

    def testDeleteSlice(self):
        count = 50
        values = [[v] for v in range(count)]
        keys = [[num2words(i)] for i in range(count)]
        remaining = sorted([k + v for k, v in zip(keys, values) if v[0] >= 40])

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.delete = tc.After(cxt.inserts, cxt.table.delete(tc.Map(views=slice(40))))
        cxt.result = tc.After(cxt.delete, cxt.table)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, remaining))

    def testUpdateSlice(self):
        count = 50
        values = [[v] for v in range(count)]
        keys = [[num2words(i)] for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table.load(SCHEMA, [k + v for k, v in zip(keys, values)])
        cxt.update = cxt.table.update({"views": 0}, {"views": slice(10)})
        cxt.result = tc.After(cxt.update, cxt.table.where({"views": slice(1)}).count())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, 10)

    def testGroupBy(self):
        count = 50
        values = [(v % 2,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.group_by(["views"]))

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [[0], [1]])

    def testOrderBy(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]
        rows = list(reversed([list(k + v) for k, v in zip(keys, values)]))

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.order_by(["views"], True))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, rows))

    def testSlice(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.where({"name": "one"}))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, [["one", 1]]))

    def testSliceAuxiliaryIndex(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.where({"views": slice(10, 20)}))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, list([[num2words(i), i] for i in range(10, 20)])))


def expected(schema, rows):
    return {str(tc.uri(tc.table.Table)): [tc.to_json(schema), rows]}


if __name__ == "__main__":
    unittest.main()
