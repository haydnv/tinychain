from __future__ import annotations

import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"
LINK = "http://127.0.0.1:8702/app/myservice"


@tc.get_op
def example(txn) -> tc.Number:
    txn.a = tc.Number(5) # this is a State
    txn.b = tc.Number(10) # this is a State
    txn.product = txn.a * txn.b # this is a Ref
    return txn.product


@tc.get_op
def to_feet(txn, meters: tc.Number) -> tc.Number:
    return tc.If(
        meters >= 0,
        meters * 3.28,
        tc.error.BadRequest("negative distance is not supported"))


class Distance(tc.Number, metaclass=tc.Meta):
    __uri__ = tc.URI(LINK) + "/distance"

    @tc.get_method
    def to_feet(self) -> Feet:
        return tc.error.NotImplemented("abstract")

    @tc.get_method
    def to_meters(self) -> Meters:
        return tc.error.NotImplemented("abstract")


class Feet(Distance):
    __uri__ = tc.URI(LINK) + "/feet"

    @tc.get_method
    def to_feet(self) -> Feet:
        return self

    @tc.get_method
    def to_meters(self) -> Meters:
        return self / 3.28


class Meters(Distance):
    __uri__ = tc.URI(LINK) + "/meters"

    @tc.get_method
    def to_feet(self) -> Feet:
        return self * 3.28

    @tc.get_method
    def to_meters(self) -> Meters:
        return self


class MyService(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI(LINK)

    def configure(self):
        self.Distance = Distance
        self.Feet = Feet
        self.Meters = Meters

    @tc.post_method
    def area(self, txn, length: Distance, width: Distance) -> Meters:
        return length.to_meters() * width.to_meters()


class MyService(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI(LINK)

    def configure(self):
        self.Distance = Distance
        self.Feet = Feet
        self.Meters = Meters

    @tc.post_method
    def area(self, txn, length: Distance, width: Distance) -> Meters:
        return length.to_meters() * width.to_meters()


class ClientService(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("http://127.0.0.1:8702/app/clientservice")

    @tc.get_method
    def room_area(self, txn, dimensions: tc.Tuple) -> Meters:
        myservice = tc.use(MyService)
        return myservice.area(length=dimensions[0], width=dimensions[1])


class ClientDocTests(unittest.TestCase):
    def setUp(self):
        self.host = start_host("test_client_docs", [MyService, ClientService])

    def testHello(self):
        hello = "Hello, World!"
        self.assertEqual(self.host.post(ENDPOINT, tc.String(hello)), hello)

    def testExampleOp(self):
        cxt = tc.Context()
        cxt.example = example
        cxt.result = cxt.example()
        self.assertEqual(self.host.post(ENDPOINT, cxt), 50)

    def tearDown(self):
        self.host.stop()


if __name__ == "__main__":
    unittest.main()

