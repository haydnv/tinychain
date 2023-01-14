import rjwt
import tinychain as tc

from ..process import DEFAULT_PORT, start_host


# TODO: define a generic class ServiceTest(unittest.TestCase) with an install_deps(for_service):... method


class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 3

    def service(self, chain_type):
        raise NotImplementedError

    def execute(self, actor, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.chain.Block)

    def testSyncChain(self):
        self._execute(tc.chain.Sync)

    def _execute(self, chain_type):
        service = self.service(chain_type)

        lead = tc.URI(service)[0]
        if lead.host() is None:
            print(f"cannot test replication of a service with no lead replica", service)
            return

        namespace = tc.URI(service).path()[1:-2]

        actor = rjwt.Actor('/')

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.URI(service).path()
            host = start_host(
                namespace,
                host_uri=host_uri,
                public_key=actor.public_key,
                cache_size=self.CACHE_SIZE,
                replicate=lead)

            hosts.append(host)

        print()
        hosts[0].create_namespace(actor, tc.URI(tc.service.Service), namespace, lead)

        print()
        hosts[0].install(actor, service)
        print()

        self.execute(actor, hosts)

        for host in hosts:
            host.stop()
