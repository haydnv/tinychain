import itertools
import random
import tinychain as tc
import unittest

from num2words import num2words
from testutils import PORT, start_host, PersistenceTest


ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.schema.Table(
    [tc.Column("name", tc.String, 512)],
    [tc.Column("views", tc.UInt)],
    {"views": ["views", "name"]})


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_table")

    def testCreate(self):
        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.result = tc.After(cxt.table.insert(("name",), (0,)), cxt.table.count())
        count = self.host.post(ENDPOINT, cxt)
        self.assertEqual(count, 1)

    def testDelete(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.delete = tc.After(cxt.inserts, cxt.table.delete())
        cxt.result = tc.After(cxt.delete, cxt.table)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected([]))

    def testDeleteSlice(self):
        count = 50
        values = [[v] for v in range(count)]
        keys = [[num2words(i)] for i in range(count)]
        remaining = sorted([k + v for k, v in zip(keys, values) if v[0] >= 40])

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.delete = tc.After(cxt.inserts, cxt.table.where(views=slice(40)).delete())
        cxt.result = tc.After(cxt.delete, cxt.table)

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(remaining))

    def testUpdateSlice(self):
        count = 50
        values = [[v] for v in range(count)]
        keys = [[num2words(i)] for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.update = tc.After(cxt.inserts, cxt.table.where(views=slice(10)).update(views=0))
        cxt.result = tc.After(cxt.update, cxt.table.where(views=slice(1)).count())

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, 10)

    def testGroupBy(self):
        count = 50
        values = [(v % 2,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.group_by(["views"]))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, {
            str(tc.uri(tc.Table)): [
                [[[], [['views', str(tc.uri(tc.UInt))]]], []],
                [[0], [1]]
            ]
        })

    def testInsert(self):
        for x in range(0, 100, 10):
            keys = list(range(x))
            random.shuffle(keys)

            cxt = tc.Context()
            cxt.table = tc.Table(SCHEMA)
            cxt.inserts = [
                cxt.table.insert((num2words(i),), (i,))
                for i in keys]

            cxt.result = tc.After(cxt.inserts, cxt.table.count())

            result = self.host.post(ENDPOINT, cxt)
            self.assertEqual(result, x)

    def testLimit(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.limit(1))

        result = self.host.post(ENDPOINT, cxt)
        first_row = sorted(list(k + v) for k, v in zip(keys, values))[0]
        self.assertEqual(result, expected([first_row]))

    def testOrderBy(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]
        rows = list(reversed([list(k + v) for k, v in zip(keys, values)]))

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.order_by(["views"], True))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(rows))

    def testSelect(self):
        count = 5
        values = [[v] for v in range(count)]
        keys = [[num2words(i)] for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.select("name"))

        expected = {
            str(tc.uri(tc.Table)): [
                tc.to_json(tc.schema.Table([tc.Column("name", tc.String, 512)])),
                list(sorted(keys))
            ]
        }

        actual = self.host.post(ENDPOINT, cxt)

        self.assertEqual(actual, expected)

    def testSlice(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.where(name="one"))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected([["one", 1]]))

    def testSliceAuxiliaryIndex(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.where(views=slice(10, 20)))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(list([[num2words(i), i] for i in range(10, 20)])))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class ChainTests(PersistenceTest, unittest.TestCase):
    NAME = "table"

    def cluster(self, chain_type):
        class Persistent(tc.Cluster):
            __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/test/table")

            def _configure(self):
                self.table = tc.chain.Block(tc.Table(SCHEMA))

        return Persistent

    def execute(self, hosts):
        row1 = {"name": "one", "views": 1}
        row2 = {"name": "two", "views": 2}

        hosts[0].put("/test/table/table", ["one"], [1])

        for host in hosts:
            actual = host.get("/test/table/table", ["one"])
            self.assertEqual(actual, row1)

        hosts[1].stop()
        hosts[2].put("/test/table/table", ["two"], [2])
        hosts[1].start()

        for host in hosts:
            actual = host.get("/test/table/table", ["one"])
            self.assertEqual(actual, row1)

            actual = host.get("/test/table/table", ["two"])
            self.assertEqual(actual, row2)

        hosts[2].stop()
        hosts[1].delete("/test/table/table", ["one"])
        hosts[2].start(wait_time=5)

        for host in hosts:
            actual = host.get("/test/table/table")
            self.assertEqual(actual, expected([["two", 2]]))

        n = 100
        for i in range(n):
            hosts[0].put("/test/table/table", [num2words(i)], [i])

        self.assertEqual(hosts[1].get("/test/table/table/count"), n)


class ErrorTest(unittest.TestCase):
    def setUp(self):
        class Persistent(tc.Cluster):
            __uri__ = tc.URI(f"/test/table")

            def _configure(self):
                self.table = tc.chain.Block(tc.Table(SCHEMA))

        self.host = start_host("table_error", [Persistent])

    def testInsert(self):
        self.assertRaises(
            tc.error.BadRequest,
            lambda: self.host.put("/test/table/table", "one", [1]))

    def tearDown(self):
        self.host.stop()


def expected(rows):
    return {str(tc.uri(tc.Table)): [tc.to_json(SCHEMA), rows]}


if __name__ == "__main__":
    unittest.main()

