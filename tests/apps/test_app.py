import unittest
import testutils
import tinychain as tc


class TestLib(tc.app.Library):
    __uri__ = tc.URI("/test/lib")

    @tc.get_method
    def up(self) -> tc.Bool:
        return True


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_lib", [TestLib()])

    def testApp(self):
        self.assertTrue(self.host.get("/test/lib/up"))


if __name__ == "__main__":
    unittest.main()
