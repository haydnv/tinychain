import typing
import unittest

import tinychain as tc

from ..process import start_host


NS = tc.URI("/test_inheritance")
LIB_NAME = "lib"
VERSION = tc.Version("0.0.0")


class Foo(tc.service.Model):
    NAME = "Foo"

    __uri__ = tc.service.model_uri(NS, LIB_NAME, VERSION, NAME)

    name = tc.String

    def __init__(self, name):
        self.name = name

    @tc.get
    def greet(self):
        return tc.String("my name is {{name}}").render(name=self.name)


class Bar(Foo):
    NAME = "Bar"

    __uri__ = tc.service.model_uri(NS, LIB_NAME, VERSION, NAME)

    @tc.get
    def greet(self):
        return tc.String("their name is {{name}}").render(name=self.name)


class Baz(Bar, tc.service.Dynamic):
    def __init__(self, name: tc.String, greetings: typing.Tuple[tc.String, ...]):
        Bar.__init__(self, name)
        self.greetings = greetings
        tc.service.Dynamic.__init__(self)

    @tc.get
    def greet(self):
        return tc.String("hello {{name}} x{{number}}").render(name=self.name, number=len(self.greetings))


class TestLib(tc.service.Library):
    NAME = LIB_NAME
    VERSION = VERSION

    __uri__ = tc.service.library_uri(None, NS, NAME, VERSION)

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
        cls.host = start_host(NS)
        cls.host.install(TestLib())

    def _get(self, endpoint):
        return self.host.get(tc.URI(TestLib).append(endpoint))

    def testApp(self):
        expected = "my name is foo"
        actual = self._get("check_foo")
        self.assertEqual(expected, actual)

        expected = "their name is bar"
        actual = self._get("check_bar")
        self.assertEqual(expected, actual)

        expected = "hello baz x3"
        actual = self._get("check_baz")
        self.assertEqual(expected, actual)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def printlines(n):
    for _ in range(n):
        print()
