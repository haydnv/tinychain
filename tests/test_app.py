import tinychain as tc
import unittest

from testutils import start_host


class TestApp(tc.Graph):
    def _schema(self):
        return tc.schema.Graph()


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_app", [TestApp], overwrite=True)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
