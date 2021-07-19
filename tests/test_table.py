import random
import tinychain as tc
import unittest

from num2words import num2words
from testutils import PORT, start_host, PersistenceTest


ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.schema.Table(
        [tc.Column("name", tc.String, 512)], [tc.Column("views", tc.UInt)]).create_index("views", ["views"])


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
        count = 2
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.delete = tc.After(cxt.inserts, cxt.table.delete())
        cxt.result = tc.After(cxt.delete, cxt.table)
        print(tc.to_json(cxt))
        return

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, []))

    @unittest.skip
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
        self.assertEqual(result, expected(SCHEMA, remaining))

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
        self.assertEqual(result, tc.to_json({
            str(tc.uri(tc.Table)): [
                [[[], [['views', tc.UInt]]], []],
                [[0], [1]]
            ]
        }))

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
        self.assertEqual(result, expected(SCHEMA, [first_row]))

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
        self.assertEqual(result, expected(SCHEMA, rows))

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
        self.assertEqual(result, expected(SCHEMA, [["one", 1]]))

    def testSliceAuxiliaryIndex(self):
        count = 50
        values = [(v,) for v in range(count)]
        keys = [(num2words(i),) for i in range(count)]

        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.inserts = [cxt.table.insert(k, v) for k, v in zip(keys, values)]
        cxt.result = tc.After(cxt.inserts, cxt.table.where(views=slice(10, 20)))

        result = self.host.post(ENDPOINT, cxt)
        self.assertEqual(result, expected(SCHEMA, list([[num2words(i), i] for i in range(10, 20)])))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class SparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_sparse_table")

    def testSlice(self):
        schema = tc.schema.Table([
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
        cxt.table = tc.Table(schema)
        cxt.inserts = [cxt.table.insert(coord, [value]) for (coord, value) in data]
        cxt.result = tc.After(cxt.inserts, cxt.table.where(**{
            "0": slice(2), "1": slice(3), "2": slice(4), "3": slice(1)
        }))

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expected(schema, [coord + [value] for coord, value in data]))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class ChainTests(PersistenceTest, unittest.TestCase):
    CACHE_SIZE = "1G"
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

        replica_set = set(str(tc.uri(host) + "/test/table") for host in hosts)
        def check_replicas():
            for i in range(len(hosts)):
                replicas = {}
                for replica in hosts[i].get("/test/table/replicas"):
                    replicas.update(replica)

                self.assertEqual(set(replicas.keys()), replica_set)

        check_replicas()

        self.assertIsNone(hosts[0].put("/test/table/table", ["one"], [1]))

        for host in hosts:
            actual = host.get("/test/table/table", ["one"])
            self.assertEqual(actual, row1)

        hosts[1].stop()
        hosts[2].put("/test/table/table", ["two"], [2])
        hosts[1].start()

        check_replicas()

        for i in range(len(hosts)):
            actual = hosts[i].get("/test/table/table", ["one"])
            self.assertEqual(actual, row1)

            actual = hosts[i].get("/test/table/table", ["two"])
            self.assertEqual(actual, row2)

        hosts[2].stop()
        self.assertIsNone(hosts[1].delete("/test/table/table", ["one"]))
        hosts[2].start()

        check_replicas()

        for i in range(len(hosts)):
            actual = hosts[i].get("/test/table/table")
            self.assertEqual(actual, expected(SCHEMA, [["two", 2]]), f"host {i}")

        self.assertIsNone(hosts[0].delete("/test/table/table"))
        for i in range(len(hosts)):
            count = hosts[i].get("/test/table/table/count")
            self.assertEqual(0, count, f"host {i}")

        total = 100
        for n in range(1, total):
            i = random.choice(range(self.NUM_HOSTS))
            print(f"host {i} insert {n}")
            self.assertIsNone(hosts[i].put("/test/table/table", [num2words(n)], [n]))

            for i in range(len(hosts)):
                count = hosts[i].get("/test/table/table/count")
                self.assertEqual(n, count, f"host {i}")


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


def expected(schema, rows):
    return {str(tc.uri(tc.Table)): [tc.to_json(schema), rows]}


if __name__ == "__main__":
    unittest.main()

