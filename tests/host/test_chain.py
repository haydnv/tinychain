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
        print(hosts[1].put("/test/chain/map", "one", tc.tensor.Dense.load([1, 2], tc.F32, [0., 0.])))


if __name__ == "__main__":
    unittest.main()
