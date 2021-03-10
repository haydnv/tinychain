import time
import tinychain as tc
import unittest

from testutils import PORT, TC_PATH, start_host


class ExampleCluster(tc.Cluster):
    __uri__ = tc.URI("/app/example")

    def _configure(self):
        self.rev = tc.Chain.Sync(tc.Number(0))


class ClusterTests(unittest.TestCase):
    def testUpdate(self):
        host = start_host("test_update", [ExampleCluster])

        def expect(n):
            actual = host.get("/app/example/rev")
            self.assertEqual(n, actual)

        expect(0)

        host.put("/app/example/rev", None, 2)
        expect(2)

        host.put("/app/example/rev", None, 4)
        expect(4)

        host.stop()
        host.start()

        actual = host.get("/app/example/rev")
        self.assertEqual(4, actual)


if __name__ == "__main__":
    unittest.main()

