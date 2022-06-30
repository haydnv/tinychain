import time
import tinychain as tc
import unittest

from ..process import DEFAULT_PORT, start_host


class Database(tc.app.App):
    __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/app/db")

    def __init__(self):
        schema = tc.table.Schema(
            [tc.Column("name", tc.String, 100)],
            [tc.Column("year", tc.UInt), tc.Column("description", tc.String, 1000)])

        self.movies = tc.chain.Block(tc.table.Table(schema))

        tc.app.App.__init__(self)

    @tc.post
    def add_movie(self, name: tc.String, year: tc.U32, description: tc.String):
        return self.movies.insert([name], [year, description])

    @tc.get
    def has_movie(self, name: tc.String):
        return self.movies.contains([name])


class Web(tc.app.App):
    __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/app/web")

    db = Database()

    def __init__(self):
        schema = tc.btree.Schema((tc.Column("name", tc.String, 100), tc.Column("views", tc.UInt)))
        self.cache = tc.chain.Sync(tc.btree.BTree(schema))
        tc.app.App.__init__(self)

    @tc.get
    def views(self, name: tc.String) -> tc.UInt:
        return self.cache[name].first()["views"]

    @tc.post
    def add_movie(self, name: tc.String, year: tc.U32, description: tc.String):
        return (
            self.db.add_movie(name=name, year=year, description=description),
            self.cache.insert([name, 0]))

    @tc.put
    def add_view(self, key: tc.String):
        # TODO: this type expectation should not be necessary
        views = tc.UInt(self.views(key)) + 1
        return tc.After(self.cache.delete(key), self.cache.insert([key, views]))


class TableDemoTests(unittest.TestCase):
    def setUp(self):
        web = Web()

        self.hosts = []
        for i in range(3):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.URI(Web).path()
            host = start_host("table_demo", web, True, host_uri, wait_time=2)
            self.hosts.append(host)

    def testCache(self):
        start = time.time()
        self.hosts[1].post("/app/web/add_movie", {"name": "Up", "year": 2009, "description": "Pixar, balloons"})
        elapsed = (time.time() - start) * 1000
        print(f"settled database transaction between two clusters of three hosts each in {elapsed:.2f}ms")

        for i in range(len(self.hosts)):
            self.assertTrue(self.hosts[i].get("/app/db/has_movie", "Up"), f"host {i} failed to update")

        self.assertEqual(self.hosts[0].get("/app/web/views", "Up"), 0)

        self.hosts[0].put("/app/web/add_view", "Up")

        for i in range(len(self.hosts)):
            self.assertEqual(self.hosts[i].get("/app/web/views", "Up"), 1, f"host {i} failed to update")

    def tearDown(self):
        for host in self.hosts:
            host.stop()
