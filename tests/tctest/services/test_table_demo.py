import time
import tinychain as tc
import unittest

from ..process import DEFAULT_PORT, start_host


LEAD = f"http://127.0.0.1:{DEFAULT_PORT}"
NS = tc.URI("/test_table_demo")


class Database(tc.service.Service):
    NAME = "db"
    VERSION = tc.Version("0.0.0")

    __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

    def __init__(self):
        schema = tc.table.Schema(
            [tc.Column("name", tc.String, 100)],
            [tc.Column("year", tc.UInt), tc.Column("description", tc.String, 1000)])

        self.movies = tc.chain.Block(tc.table.Table(schema))

        tc.service.Service.__init__(self)

    @tc.post
    def add_movie(self, name: tc.String, year: tc.U32, description: tc.String):
        return self.movies.insert([name], [year, description])

    @tc.get
    def has_movie(self, name: tc.String):
        return self.movies.contains([name])


class Web(tc.service.Service):
    NAME = "web"
    VERSION = tc.Version("0.0.0")

    __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

    db = Database()

    def __init__(self):
        schema = tc.btree.Schema((tc.Column("name", tc.String, 100), tc.Column("views", tc.UInt)))
        self.cache = tc.chain.Sync(tc.btree.BTree(schema))
        tc.service.Service.__init__(self)

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
        views = self.views(key) + 1
        return tc.after(self.cache.delete(key), self.cache.insert([key, views]))


class TableDemoTests(unittest.TestCase):
    def setUp(self):
        db = Database()
        web = Web()

        self.hosts = []
        for i in range(3):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.URI(Web).path()
            host = start_host(NS, host_uri=host_uri, replicate=LEAD)
            self.hosts.append(host)

        self.hosts[0].put(tc.URI(tc.service.Service), str(NS)[1:], tc.URI(web)[:-2])
        self.hosts[0].install(db)
        self.hosts[0].install(web)

    def testCache(self):
        db = tc.URI(Database).path()
        web = tc.URI(Web).path()

        start = time.time()
        self.hosts[1].post(web.append("add_movie"), {"name": "Up", "year": 2009, "description": "Pixar, balloons"})
        elapsed = (time.time() - start) * 1000

        print(f"settled database transaction between two clusters of three hosts each in {elapsed:.2f}ms")

        for i in range(len(self.hosts)):
            self.assertTrue(self.hosts[i].get(db.append("has_movie"), "Up"), f"host {i} failed to update")

        self.assertEqual(self.hosts[0].get(web.append("views"), "Up"), 0)

        self.hosts[0].put(web.append("add_view"), "Up")

        for i in range(len(self.hosts)):
            self.assertEqual(self.hosts[i].get(web.append("views"), "Up"), 1, f"host {i} failed to update")

    def tearDown(self):
        for host in self.hosts:
            host.stop()
