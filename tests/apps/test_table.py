import random
import tinychain as tc
import unittest

from num2words import num2words
from testutils import DEFAULT_PORT, start_docker, PersistenceTest

ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.table.Schema(
    [tc.Column("name", tc.String, 512)], [tc.Column("views", tc.UInt)]).create_index("views", ["views"])


class TableChainTests(PersistenceTest, unittest.TestCase):
    NAME = "table"
    NUM_HOSTS = 4

    def app(self, chain_type):
        class Persistent(tc.app.App):
            __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/test/table")

            def __init__(self):
                self.table = chain_type(tc.table.Table(SCHEMA))
                tc.app.App.__init__(self)

            @tc.delete_method
            def truncate(self):
                return self.table.delete()

        return Persistent()

    def execute(self, hosts):
        row1 = ["one", 1]
        row2 = ["two", 2]

        self.assertIsNone(hosts[0].put("/test/table/table", ["one"], [1]))

        for host in hosts:
            actual = host.get("/test/table/table", ["one"])
            self.assertEqual(actual, row1)

        hosts[1].stop()
        hosts[2].put("/test/table/table", ["two"], [2])
        hosts[1].start()

        for i in range(len(hosts)):
            actual = hosts[i].get("/test/table/table", ["one"])
            self.assertEqual(actual, row1)

            actual = hosts[i].get("/test/table/table", ["two"])
            self.assertEqual(actual, row2)

        hosts[2].stop()
        self.assertIsNone(hosts[1].delete("/test/table/table", ["one"]))
        hosts[2].start()

        for i in range(len(hosts)):
            actual = hosts[i].get("/test/table/table")
            self.assertEqual(actual, expected(SCHEMA, [["two", 2]]), f"host {i}")

        self.assertIsNone(hosts[0].delete("/test/table/truncate"))
        for i in range(len(hosts)):
            count = hosts[i].get("/test/table/table/count")
            self.assertEqual(0, count, f"host {i}")

        total = 100
        for n in range(1, total):
            i = random.choice(range(self.NUM_HOSTS))

            self.assertIsNone(hosts[i].put("/test/table/table", [num2words(n)], [n]))

            for i in range(len(hosts)):
                count = hosts[i].get("/test/table/table/count")
                self.assertEqual(n, count, f"host {i}")


class TableErrorTest(unittest.TestCase):
    def setUp(self):
        class Persistent(tc.app.App):
            __uri__ = tc.URI(f"/test/table")

            def __init__(self):
                self.table = tc.chain.Block(tc.table.Table(SCHEMA))
                tc.app.App.__init__(self)

        self.host = start_docker("test_table_error", [Persistent()])

    def testInsert(self):
        self.assertRaises(
            tc.error.BadRequest,
            lambda: self.host.put("/test/table/table", "one", [1]))

    def tearDown(self):
        self.host.stop()


def expected(schema, rows):
    return {str(tc.uri(tc.table.Table)): [tc.to_json(schema), rows]}


if __name__ == "__main__":
    unittest.main()
