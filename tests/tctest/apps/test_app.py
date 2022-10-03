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


class TestApp(tc.app.App):
    __uri__ = URI

    User = User

    @tc.post
    def create_user(self, _first_name: tc.String, _last_name: tc.String):
        return tc.error.NotImplemented("create user")


class UserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_lib", [TestApp()])

    def testCreateUser(self):
        self.assertRaises(tc.error.NotImplemented, lambda: self.host.post("/test/app/create_user"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
