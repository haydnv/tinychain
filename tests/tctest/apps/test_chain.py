import unittest
import tinychain as tc

from .base import PersistenceTest
from ..process import DEFAULT_PORT


class MapChainTests(PersistenceTest, unittest.TestCase):
    NAME = "chain"

    def app(self, chain_type):
        class Persistent(tc.app.App):
            __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/test/chain")

            def __init__(self):
                self.map = chain_type(tc.Map({}))
                tc.app.App.__init__(self)

        return Persistent()

    def execute(self, hosts):
        hosts[1].put("/test/chain/map", "one", load_dense([1, 2], tc.F32, [0., 0.]))

        for i in range(len(hosts)):
            host = hosts[i]
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 0)

        hosts[2].put("/test/chain/map/one", value=load_dense([1, 2], tc.F32, [1., 1.]))

        for host in hosts:
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 2)

        hosts[3].stop()
        hosts[2].put("/test/chain/map/one", value=load_dense([1, 2], tc.F32, [0.1, 0.9]))
        hosts[3].start()

        for host in hosts:
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 1)

        hosts[3].put("/test/chain/map/one", value=load_dense([1, 2], tc.F32, [1.5, 2.5]))

        for host in hosts:
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 4)


def load_dense(shape, dtype, elements):
    return {str(tc.URI(tc.tensor.Dense)): [[shape, dtype], elements]}
