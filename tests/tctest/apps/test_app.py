import tinychain as tc
import unittest

from ..process import start_host

URI = tc.URI("/test/app")


class User(tc.app.Model):
    __uri__ = URI.append("Foo")

    first_name = tc.Column(tc.String, 100)
    last_name = tc.Column(tc.String, 100)
    email = tc.Column(tc.EmailAddress, 100)

    def __init__(self, first_name, last_name, email):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email


class TestApp(tc.app.App):
    __uri__ = URI

    @tc.post
    def create_user(self, new_user: User):
        """TODO"""


class UserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_lib", [TestApp()])

    def testCreateUser(self):
        details = {"first_name": "First", "last_name": "Last", "email": "email@example.com"}
        self.host.post(tc.URI(TestApp).append("create_user"), details)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
