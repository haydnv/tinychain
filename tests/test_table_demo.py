import tinychain as tc
import unittest

from testutils import PORT, start_host


class Database(tc.Cluster):
    __uri__ = tc.URI("/app/db")

    def _configure(self):
        schema = tc.schema.Table(
            [tc.Column("name", tc.String, 100)],
            [tc.Column("year", tc.UInt), tc.Column("description", tc.String, 1000)])

        self.movies = tc.Chain.Block(tc.Table(schema))


class Web(tc.Cluster):
    __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/app/web")

    def _configure(self):
        schema = tc.schema.BTree(tc.Column("name", tc.String, 100), tc.Column("views", tc.UInt))
        self.cache = tc.Chain.Sync(tc.BTree(schema))

    @tc.get_method
    def views(self, txn, name: tc.String) -> tc.UInt:
        return self.cache[name].first()["views"]

    @tc.put_method
    def add_movie(self, txn, name: tc.String, metadata: tc.Map):
        db = tc.use(Database)

        return (
            db.movies.insert([name], [metadata["year"], metadata["description"]]),
            self.cache.insert([name, 0]))

    @tc.post_method
    def add_view(self, txn, name: tc.String):
        txn.views = self.views(name)
        return tc.After(
            self.cache[name, txn.views].delete(),
            self.cache.insert([name, txn.views + 1]))


@unittest.skip
class DemoTests(unittest.TestCase):
    def setUp(self):
        self.hosts = []
        for i in range(3):
            port = PORT + i
            host_uri = tc.URI(f"http://127.0.0.1:{port}") + tc.uri(Web).path()
            host = start_host("table_demo", [Database, Web], True, host_uri)
            self.hosts.append(host)

    def testCache(self):
        self.hosts[1].put("/app/web/add_movie",
            "Up", {"year": 2009, "description": "Pixar, balloons"})

        self.assertEqual(self.hosts[2].get("/app/web/views", "Up"), 0)

        for i in range(5):
            print()

        self.hosts[0].post("/app/web/add_view", {"name": "Up"})
        self.assertEqual(self.hosts[1].get("/app/web/views", "Up"), 1)

    def tearDown(self):
        for host in self.hosts:
            host.stop()


if __name__ == "__main__":
    unittest.main()

