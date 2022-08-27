import os
import tinychain as tc
import unittest

TC_HOST = os.getenv("TC_HOST", "http://localhost:8702")


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = tc.host.Host(TC_HOST)
