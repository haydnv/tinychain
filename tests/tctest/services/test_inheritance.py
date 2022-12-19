import typing
import unittest
import tinychain as tc

from ..process import start_host


NS = tc.URI("/test_inheritance")
NAME = "test_lib"
VERSION = tc.Version("0.0.0")


class Foo(tc.app.Model):
    NS = NS.append(NAME)
    NAME = "Foo"

    # TODO: replace URI("/class") with URI(tc.app.Model)
    __uri__ = (tc.URI("/class") + NS).extend(VERSION, NAME)

    name = tc.String

    def __init__(self, name):
        self.name = name

    @tc.get
    def greet(self):
        return tc.String("my name is {{name}}").render(name=self.name)


class Bar(Foo):
    NS = NS.append(NAME)
    NAME = "Bar"

    # TODO: replace URI("/class") with URI(tc.app.Model)
    __uri__ = (tc.URI("/class") + NS).extend(VERSION, NAME)

    @tc.get
    def greet(self):
        return tc.String("their name is {{name}}").render(name=self.name)


class Baz(Bar, tc.app.Dynamic):
    def __init__(self, name: tc.String, greetings: typing.Tuple[tc.String, ...]):
        Bar.__init__(self, name)
        self.greetings = greetings
        tc.app.Dynamic.__init__(self)

    @tc.get
    def greet(self):
        return tc.String("hello {{name}} x{{number}}").render(name=self.name, number=len(self.greetings))


class TestLib(tc.app.Library):
    NS = NS
    NAME = NAME
    VERSION = VERSION

    __uri__ = (tc.URI(tc.app.Library) + NS).extend(NAME, VERSION)

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
        cls.host = start_host(str(NS)[1:], [])
        cls.host.put(tc.URI(tc.app.Library), "test_inheritance", tc.URI(tc.app.Library) + NS)
        cls.host.put(tc.URI(TestLib)[:-2], NAME, TestLib())

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
