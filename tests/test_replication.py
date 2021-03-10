import tinychain as tc
import unittest

from testutils import PORT, start_host


NUM_HOSTS = 4


class Rev(tc.Cluster):
    __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/app/test/replication")

    def _configure(self):
        self.rev = tc.Chain.Sync(0)

    @tc.get_method
    def version(self) -> tc.Number:
        return self.rev

    @tc.post_method
    def bump(self, txn):
        txn.rev = self.version()
        return self.rev.set(txn.rev + 1)


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        hosts = []
        for i in range(NUM_HOSTS):
            port = PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.uri(Rev).path()
            host = start_host(f"test_replication_{i}", [Rev], host_uri=tc.URI(host_uri))
            hosts.append(host)
            printlines(5)

        self.hosts = hosts

    def testReplication(self):
        cluster_path = tc.uri(Rev).path()

        expected = set("http://" + tc.uri(host) + cluster_path for host in self.hosts)
        for host in self.hosts:
            actual = {}
            for link in host.get(cluster_path + "/replicas"):
                actual.update(link)
            actual = set(actual.keys())

            self.assertEqual(expected, actual)

    def tearDown(self):
        for host in self.hosts:
            host.stop()


def printlines(n):
    for _ in range(n):
        print()

if __name__ == "__main__":
    unittest.main()

