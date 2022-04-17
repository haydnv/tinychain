import os
import tinychain as tc
import unittest


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = tc.host.Host(os.getenv("TC_HOST", "http://demo.tinychain.net"))
