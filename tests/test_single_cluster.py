import tinychain as tc
import unittest

from testutils import start_host


class BlockChainTest(tc.Cluster):
    __uri__ = tc.URI("/app/example/block")

    def _configure(self):
        self.rev = tc.chain.Block(tc.Number(0))


class SyncChainTest(tc.Cluster):
    __uri__ = tc.URI("/app/example/sync")

    def _configure(self):
        self.rev = tc.chain.Sync(tc.Number(0))


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.host = start_host("test_update", [BlockChainTest, SyncChainTest])

    def _test(self, endpoint):
        def expect(n):
            actual = self.host.get(endpoint)
            self.assertEqual(n, actual)

        expect(0)

        self.host.put(endpoint, None, 2)
        expect(2)

        self.host.put(endpoint, None, 4)
        expect(4)

        self.host.stop()
        self.host.start()

        actual = self.host.get(endpoint)
        self.assertEqual(4, actual)

    def testBlockChain(self):
        self._test("/app/example/block/rev")

    def testSyncChain(self):
        self._test("/app/example/sync/rev")

    def tearDown(self):
        self.host.stop()


if __name__ == "__main__":
    unittest.main()

