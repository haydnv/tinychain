import unittest
import testutils
import tinychain as tc


class App(tc.Cluster):
    __uri__ = tc.URI("/test/app")

    @tc.get_method
    def up(self) -> tc.Bool:
        return True


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_app", [App], wait_time=2.)

    def testApp(self):
        self.assertTrue(self.host.get("/test/app/up"))



if __name__ == "__main__":
    unittest.main()
