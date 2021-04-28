import tinychain as tc
import unittest

from testutils import start_host, PORT


ENDPOINT = "/transact/hypothetical"
SCHEMA = tc.Table.Schema(
    [tc.Column("name", tc.String, 512)],
    [tc.Column("views", tc.UInt)])


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_table")

    def testCreate(self):
        cxt = tc.Context()
        cxt.table = tc.Table(SCHEMA)
        cxt.result = tc.After(cxt.table.insert(("Movie Name",), (0,)), cxt.table.count())

        count = self.host.post(ENDPOINT, cxt)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()

