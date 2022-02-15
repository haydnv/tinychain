import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/lib")


class Foo(tc.app.Model):
    __uri__ = URI.append("Foo")

    name = tc.String

    def __init__(self, name):
        self.name = name

    @tc.get_method
    def greet(self):
        return tc.String("my name is {{name}}").render(name=self.name)


class TestLib(tc.app.Library):
    __uri__ = URI

    def exports(self):
        return [Foo]

    @tc.get_method
    def check_model(self, cxt) -> Foo:
        cxt.foo = self.Foo("foo")
        return cxt.foo.greet()


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_lib", [TestLib()])

    def testApp(self):
        expected = "my name is foo"
        actual = self.host.get("/test/lib/check_model")
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
