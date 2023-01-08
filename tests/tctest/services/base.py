import tinychain as tc

from ..process import DEFAULT_PORT, start_host


class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 3

    def service(self, chain_type):
        raise NotImplementedError

    def execute(self, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.chain.Block)

    def testSyncChain(self):
        self._execute(tc.chain.Sync)

    def _execute(self, chain_type):
        service = self.service(chain_type)

        lead = tc.URI(service)[0]
        if "://" not in lead:
            print(f"cannot test replication of a service with no lead replica", service)
            return

        [namespace] = tc.URI(service).path()[1:-2]

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.URI(service).path()
            host = start_host(tc.URI(f"/{namespace}"), host_uri=host_uri, cache_size=self.CACHE_SIZE, replicate=str(lead))
            hosts.append(host)

        print()
        hosts[0].put(tc.URI(tc.service.Service), namespace, tc.URI(service)[:-2])

        print()
        hosts[0].put(tc.URI(service).path()[:-2], tc.URI(service)[-2], service)
        print()

        self.execute(hosts)

        for host in hosts:
            host.stop()
