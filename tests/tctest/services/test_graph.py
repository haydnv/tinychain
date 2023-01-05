import unittest

import tinychain as tc

from ..process import DEFAULT_PORT, start_host


LEAD = f"http://127.0.0.1:{DEFAULT_PORT}"


class User(tc.app.Model):
    NS = tc.URI("/test/graph")
    NAME = "User"
    VERSION = tc.Version("0.0.0")

    # TODO: replace URI("/class") with URI(tc.app.Model)
    __uri__ = (tc.URI("/class") + NS).extend(VERSION, NAME)

    first_name = tc.Column("first_name", tc.String, 100)
    last_name = tc.Column("last_name", tc.String, 100)

    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name


class TestService(tc.graph.Graph):
    HOST = tc.URI(LEAD)
    NS = tc.URI("/test")
    NAME = "graph"
    VERSION = tc.Version("0.0.0")

    User = User

    __uri__ = HOST + tc.URI(tc.app.Service) + NS.extend(NAME, VERSION)

    @tc.post
    def create_user(self, first_name: tc.String, last_name: tc.String):
        # TODO: change method signature to (self, new_user: User) on completion of #175
        user_id = self.user.max_id() + 1
        return tc.after(self.user.insert([user_id], [first_name, last_name]), user_id)


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_app", [])

        cls.host.put(tc.URI(tc.app.Service), str(TestService.NS)[1:], tc.URI(TestService)[:-2])
        cls.host.install(TestService())

    def testCreateUser(self):
        uri = tc.URI(TestService).path()
        count = self.host.get(uri.extend("user", "count"))
        user_id = self.host.post(uri.append("create_user"), {"first_name": "First", "last_name": "Last"})
        self.assertEqual(self.host.get(uri.extend("user", "count")), count + 1)
        self.assertEqual(self.host.get(uri.append("user"), [user_id]), [user_id, "First", "Last"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
