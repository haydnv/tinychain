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


class Bar(Foo):
    __uri__ = URI.append("Bar")

    @tc.get_method
    def greet(self):
        return tc.String("their name is {{name}}").render(name=self.name)


class TestLib(tc.app.Library):
    __uri__ = URI

    def exports(self):
        return [Foo, Bar]

    @tc.get_method
    def check_foo(self, cxt) -> Foo:
        cxt.foo = self.Foo("foo")
        return cxt.foo.greet()

    @tc.get_method
    def check_bar(self, cxt) -> Foo:
        cxt.bar = self.Bar("bar")
        return cxt.bar.greet()


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_lib", [TestLib()])

    def testApp(self):
        expected = "my name is foo"
        actual = self.host.get("/test/lib/check_foo")
        self.assertEqual(expected, actual)

        expected = "their name is bar"
        actual = self.host.get("/test/lib/check_bar")
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
