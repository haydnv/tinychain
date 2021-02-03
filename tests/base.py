import unittest

import tinychain as tc


PATH = "../host/target/debug/tinychain"
PORT = 8702


class InstanceTest(unittest.TestCase):
    def setUp(self):
        self.host = tc.host.Local(PATH, "tctest_examples", PORT, log_level="debug")

    def tearDown(self):
        self.host.terminate()
