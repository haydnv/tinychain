import unittest

import tinychain as tc

from ..process import start_host

URI = tc.URI("/test/app")


class User(tc.app.Model):
    __uri__ = URI.append("user")

    first_name = tc.Column("first_name", tc.String, 100)
    last_name = tc.Column("last_name", tc.String, 100)

    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name


class TestApp(tc.graph.Graph):
    __uri__ = URI

    @tc.post
    def create_user(self, first_name: tc.String, last_name: tc.String):
        # TODO: change method signature to (self, new_user: User) on completion of #175
        user_id = self.user.max_id() + 1
        return tc.after(self.user.insert([user_id], [first_name, last_name]), user_id)


class UserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_app", [TestApp.autogenerate([User])])

    def testCreateUser(self):
        count = self.host.get(URI.append("user/count"))
        user_id = self.host.post(
            URI.append("create_user"), {"first_name": "First", "last_name": "Last"}
        )
        self.assertEqual(self.host.get(URI.append("user/count")), count + 1)
        self.assertEqual(self.host.get(URI.append("user"), [user_id]), [user_id, "First", "Last"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
