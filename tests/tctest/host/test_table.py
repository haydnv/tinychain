import random
import tinychain as tc
import unittest

from num2words import num2words
from .base import HostTest

ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.table.Schema(
    [tc.Column("name", tc.String, 512)], [tc.Column("views", tc.UInt)]).create_index("views", ["views"])


class TableTests(HostTest):
    def testCreate(self):
        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, []))

    def testDelete(self):
        count = 10
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.delete = tc.after(cxt.inserts, cxt.table.delete_row(("one",)))
        cxt.result = tc.after(cxt.delete, cxt.table.count())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, count - 1)

    def testInsert(self):
        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.upsert = cxt.table.upsert((num2words(1),), (1,))
        cxt.result = tc.after(cxt.upsert, cxt.table.count())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, 1)

        for x in range(0, 20, 5):
            keys = list(range(x))
            random.shuffle(keys)

            cxt = tc.Context()
            cxt.table = tc.table.Table(SCHEMA)
            cxt.inserts = [
                cxt.table.insert((num2words(i),), (i,))
                for i in keys]

            cxt.result = tc.after(cxt.inserts, cxt.table.count())

            result = self.host.post(ENDPOINT, cxt)
            self.assertEqual(result, x)

    def testLimit(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.after(cxt.inserts, cxt.table.limit(1))

        result = self.host.post(ENDPOINT, cxt)
        first_row = sorted(list(k + v) for k, v in zip(keys, values))[0]
        self.assertEqual(result, expected(SCHEMA, [first_row]))

    def testSelect(self):
        count = 5
        values = [[v] for v in range(count)]
        keys = [[num2words(i)] for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.after(cxt.inserts, cxt.table.select(["name"]))

        expected = {
            str(tc.URI(tc.table.Table)): [
                tc.to_json(tc.table.Schema([], [tc.Column("name", tc.String, 512)])),
                list(sorted(keys))
            ]
        }

        actual = self.host.post(ENDPOINT, cxt)

        self.assertEqual(actual, expected)


def expected(schema, rows):
    return {str(tc.URI(tc.table.Table)): [tc.to_json(schema), rows]}


if __name__ == "__main__":
    unittest.main()
