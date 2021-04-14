import testutils
import tinychain as tc
import unittest


ENDPOINT = "/transact/hypothetical"


class BTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_btree")

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()

