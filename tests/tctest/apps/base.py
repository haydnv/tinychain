import time
import tinychain as tc

from ..process import DEFAULT_PORT, start_host


class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 4
    NAME = "persistence"

    def app(self, chain_type):
        raise NotImplementedError

    def execute(self, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.chain.Block)

    def testSyncChain(self):
        self._execute(tc.chain.Sync)

    def _execute(self, chain_type):
        name = self.NAME

        app = self.app(chain_type)

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.uri(app).path()
            host = start_host(f"test_{name}_{i}", [app], host_uri=tc.URI(host_uri), cache_size=self.CACHE_SIZE)
            hosts.append(host)

        time.sleep(1)

        self.execute(hosts)

        for host in hosts:
            host.stop()
