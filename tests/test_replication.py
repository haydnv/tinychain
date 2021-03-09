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

        self.hosts = hosts

    def testReplication(self):
        for host in self.hosts:
            rev = host.get(tc.uri(Rev).path() + "/version")
            self.assertEqual(rev, 0)

    def tearDown(self):
        for host in self.hosts:
            host.stop()


if __name__ == "__main__":
    unittest.main()

