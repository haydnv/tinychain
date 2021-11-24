import unittest
import testutils
import tinychain as tc


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host()

    def testApp(self):
        self.assertRaises(tc.error.NotFound, lambda: self.host.get("/test/app"))


if __name__ == "__main__":
    unittest.main()
