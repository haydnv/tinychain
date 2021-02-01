import os
import sys
import unittest

import tinychain as tc


PATH = os.environ.get('TC_PATH')
if not PATH:
    sys.exit("Required environment variable not set: TC_PATH")

PORT = 8702


class InstanceTest(unittest.TestCase):
    def setUp(self):
        self.host = tc.host.Local(PATH, "tctest_examples", PORT, log_level="debug")

    def tearDown(self):
        self.host.terminate()
