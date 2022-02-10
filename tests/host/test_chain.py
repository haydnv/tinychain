import unittest
import tinychain as tc

from testutils import DEFAULT_PORT, PersistenceTest


class ChainTests(PersistenceTest, unittest.TestCase):
    NAME = "chain"

    def cluster(self, chain_type):
        class Persistent(tc.Cluster, metaclass=tc.Meta):
            __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/test/chain")

            def _configure(self):
                self.map = chain_type(tc.Map({}))

        return Persistent

    def execute(self, hosts):
        hosts[1].put("/test/chain/map", "one", tc.tensor.Dense.load([1, 2], tc.F32, [0., 0.]))

        for i in range(len(hosts)):
            host = hosts[i]
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 0)

        hosts[2].put("/test/chain/map/one", value=tc.tensor.Dense.load([1, 2], tc.F32, [1., 1.]))

        for host in hosts:
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 2)

        hosts[3].stop()
        hosts[2].put("/test/chain/map/one", value=tc.tensor.Dense.load([1, 2], tc.F32, [0.1, 0.9]))
        hosts[3].start()

        for host in hosts:
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 1)

        hosts[3].put("/test/chain/map/one", value=tc.tensor.Dense.load([1, 2], tc.F32, [1.5, 2.5]))

        for host in hosts:
            sum = host.get("/test/chain/map/one/sum")
            self.assertEqual(sum, 4)


if __name__ == "__main__":
    unittest.main()
