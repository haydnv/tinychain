import tinychain as tc
import unittest

from testutils import start_host


class Database(tc.Cluster):
    __uri__ = tc.URI("/app/db")

    def _configure(self):
        schema = tc.Table.Schema(
            [tc.Column("name", tc.String, 100)],
            [tc.Column("year", tc.UInt), tc.Column("description", tc.String, 1000)])

        self.movies = tc.Chain.Block(tc.Table(schema))


class Web(tc.Cluster):
    __uri__ = tc.URI("/app/web")

    def _configure(self):
        schema = tc.BTree.Schema(tc.Column("name", tc.String, 100), tc.Column("views", tc.UInt))
        self.cache = tc.Chain.Sync(tc.BTree(schema))

    @tc.get_method
    def views(self, txn, name: tc.String):
        txn.slice = self.cache[name]
        txn.row = txn.slice.first()
        return txn.row["views"]

    @tc.put_method
    def add_movie(self, txn, name: tc.String, metadata: tc.Map):
        db = tc.use(Database)

        return (
            db.movies.insert([name], [metadata["year"], metadata["description"]]),
            self.cache.insert([name, 0]))

class DemoTests(unittest.TestCase):
    def setUp(self):
        self.host = start_host("table_demo", [Database, Web], True)

    def testCache(self):
        self.host.put("/app/web/add_movie", "Up", {"year": 2009, "description": "Pixar, balloons"})
        self.assertEqual(self.host.get("/app/web/views", "Up"), 0)

    def tearDown(self):
        self.host.stop()


if __name__ == "__main__":
    unittest.main()

