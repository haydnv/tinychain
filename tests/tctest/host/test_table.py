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
        cxt.result = tc.after(cxt.table.insert(("name",), (0,)), cxt.table.count())

        count = self.host.post(ENDPOINT, cxt)
        self.assertEqual(count, 1)

    def testDelete(self):
        count = 2
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.table.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.delete = tc.after(cxt.inserts, cxt.table.delete())
        cxt.result = tc.after(cxt.delete, cxt.table)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, []))

    def testInsert(self):
        for x in range(0, 100, 10):
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
                tc.to_json(tc.table.Schema([tc.Column("name", tc.String, 512)])),
                list(sorted(keys))
            ]
        }

        actual = self.host.post(ENDPOINT, cxt)

        self.assertEqual(actual, expected)


class SparseTests(HostTest):
    def testSlice(self):
        schema = tc.table.Schema([
            tc.Column("0", tc.U64),
            tc.Column("1", tc.U64),
            tc.Column("2", tc.U64),
            tc.Column("3", tc.U64),
        ], [
            tc.Column("value", tc.Number),
        ])

        for i in range(4):
            schema.create_index(str(i), [str(i)])

        data = [
            ([0, 0, 1, 0], 1),
            ([0, 1, 2, 0], 2),
            ([1, 0, 0, 0], 3),
            ([1, 0, 1, 0], 3),
        ]

        cxt = tc.Context()
        cxt.table = tc.table.Table(schema)
        cxt.inserts = [cxt.table.insert(coord, [value]) for (coord, value) in data]
        cxt.result = tc.after(cxt.inserts, cxt.table.where({
            "0": slice(2),
            "1": slice(3),
            "2": slice(4),
            "3": slice(1)
        }))

        expect = expected(schema, [coord + [value] for coord, value in data])
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


def expected(schema, rows):
    return {str(tc.URI(tc.table.Table)): [tc.to_json(schema), rows]}


if __name__ == "__main__":
    unittest.main()
