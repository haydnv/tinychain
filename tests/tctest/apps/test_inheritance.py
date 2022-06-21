import typing
import unittest
import tinychain as tc

from ..process import start_host


URI = tc.URI("/test/lib")


class Foo(tc.app.Model):
    __uri__ = URI.append("Foo")

    name = tc.String

    def __init__(self, name):
        self.name = name

    @tc.get
    def greet(self):
        return tc.String("my name is {{name}}").render(name=self.name)


class Bar(Foo):
    __uri__ = URI.append("Bar")

    @tc.get
    def greet(self):
        return tc.String("their name is {{name}}").render(name=self.name)


class Baz(Bar, tc.app.Dynamic):
    def __init__(self, name: tc.String, greetings: typing.Tuple[tc.String, ...]):
        Bar.__init__(self, name)
        self.greetings = greetings

    @tc.get
    def greet(self):
        return tc.String("hello {{name}} x{{number}}").render(name=self.name, number=len(self.greetings))


class TestLib(tc.app.Library):
    __uri__ = URI

    Foo = Foo
    Bar = Bar
    Baz = Baz

    @tc.get
    def check_foo(self, cxt) -> tc.String:
        cxt.foo = self.Foo("foo")
        return cxt.foo.greet()

    @tc.get
    def check_bar(self, cxt) -> tc.String:
        cxt.bar = self.Bar("bar")
        return cxt.bar.greet()

    @tc.get
    def check_baz(self, cxt) -> tc.String:
        cxt.baz = self.Baz("baz", ["one", "two", "three"])
        return cxt.baz.greet()


class InheritanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_inheritance", [TestLib()])

    def testApp(self):
        expected = "my name is foo"
        actual = self.host.get("/test/lib/check_foo")
        self.assertEqual(expected, actual)

        expected = "their name is bar"
        actual = self.host.get("/test/lib/check_bar")
        self.assertEqual(expected, actual)

        expected = "hello baz x3"
        actual = self.host.get("/test/lib/check_baz")
        self.assertEqual(expected, actual)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()
