import tinychain as tc
import unittest

from testutils import DEFAULT_PORT, start_host


class Database(tc.Cluster):
    __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/app/db")

    def _configure(self):
        schema = tc.table.Schema(
            [tc.Column("name", tc.String, 100)],
            [tc.Column("year", tc.UInt), tc.Column("description", tc.String, 1000)])

        self.movies = tc.chain.Block(tc.table.Table(schema))

    @tc.get_method
    def has_movie(self, name: tc.String):
        return self.movies.contains([name])


class Web(tc.Cluster):
    __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/app/web")

    def _configure(self):
        schema = tc.btree.Schema((tc.Column("name", tc.String, 100), tc.Column("views", tc.UInt)))
        self.cache = tc.chain.Sync(tc.btree.BTree(schema))

    @tc.get_method
    def views(self, name: tc.String) -> tc.UInt:
        return self.cache[name].first()["views"]

    @tc.post_method
    def add_movie(self, name: tc.String, year: tc.U32, description: tc.String):
        db = tc.use(Database)

        return (
            db.movies.insert([name], [year, description]),
            self.cache.insert([name, 0]))

    @tc.put_method
    def add_view(self, txn, key: tc.String):
        txn.views = self.views(key)
        return tc.After(
            self.cache.delete(key),
            self.cache.insert([key, txn.views + 1]))


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.hosts = []
        for i in range(3):
            port = DEFAULT_PORT + i
            host_uri = tc.URI(f"http://127.0.0.1:{port}") + tc.uri(Web).path()
            host = start_host("table_demo", [Database, Web], True, host_uri, wait_time=2)
            self.hosts.append(host)

    def testCache(self):
        self.hosts[1].post("/app/web/add_movie", {"name": "Up", "year": 2009, "description": "Pixar, balloons"})

        for i in range(len(self.hosts)):
            self.assertTrue(self.hosts[i].get("/app/db/has_movie", "Up"), f"host {i} failed to update")

        self.assertEqual(self.hosts[0].get("/app/web/views", "Up"), 0)

        self.hosts[0].put("/app/web/add_view", "Up")

        for i in range(len(self.hosts)):
            self.assertEqual(self.hosts[i].get("/app/web/views", "Up"), 1, f"host {i} failed to update")

    def tearDown(self):
        for host in self.hosts:
            host.stop()


if __name__ == "__main__":
    unittest.main()
