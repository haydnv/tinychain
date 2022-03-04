import logging
import os
import pathlib
import shutil
import subprocess
import time
import tinychain as tc


TC_PATH = os.getenv("TC_PATH", "host/target/debug/tinychain")
DEFAULT_PORT = 8702


class Process(tc.host.Local.Process):
    """A local TinyChain host process."""

    ADDRESS = "127.0.0.1"

    def __init__(self, path, workspace, force_create=False,
                 data_dir=None, clusters=[], port=DEFAULT_PORT, log_level="warn", cache_size="1G", request_ttl="30"):

        print(f"start host process on port {port}")

        # set _process first so it's available to __del__ in case of an exception
        self._process = None

        if port is None or not int(port) or int(port) < 0:
            raise ValueError(f"invalid port: {port}")

        if clusters and data_dir is None:
            raise ValueError("Hosting a cluster requires specifying a data_dir")

        if data_dir:
            maybe_create_dir(data_dir, force_create)

        args = [
            path,
            f"--workspace={workspace}",
            f"--address={self.ADDRESS}",
            f"--http_port={port}",
            f"--log_level={log_level}",
            f"--cache_size={cache_size}",
            f"--request_ttl={request_ttl}",
        ]

        if data_dir:
            args.append(f"--data_dir={data_dir}")

        args.extend([f"--cluster={cluster}" for cluster in clusters])

        self._args = args

    def start(self, wait_time):
        """Start this host `Process`."""

        if self._process:
            raise RuntimeError("tried to start a host that's already running")

        self._process = subprocess.Popen(self._args)
        time.sleep(wait_time)

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(f"TinyChain process crashed on startup")
        else:
            logging.info(f"new instance running")

    def stop(self, wait_time=None):
        """Shut down this host `Process`."""

        logging.info(f"Shutting down TinyChain host")
        if self._process:
            self._process.terminate()
            self._process.wait()
            logging.info(f"Host shut down")

            if wait_time:
                time.sleep(wait_time)
        else:
            logging.info(f"Host not running")

        self._process = None

    def __del__(self):
        if self._process:
            self.stop()


def start_host(name, clusters=[], overwrite=True, host_uri=None, cache_size="5K", wait_time=1, timeout=30):
    if not os.path.isfile(TC_PATH):
        raise RuntimeError(f"invalid executable path: {TC_PATH}")

    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()
    elif clusters and tc.uri(clusters[0]).port():
        port = tc.uri(clusters[0]).port()

    config = []
    for cluster in clusters:
        cluster_config = f"config/{name}"
        cluster_config += str(tc.uri(cluster).path())

        tc.write_cluster(cluster, cluster_config, overwrite)
        config.append(cluster_config)

    data_dir = f"/tmp/tc/tmp/{port}/{name}"
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    process = Process(
        TC_PATH,
        workspace=f"/tmp/tc/tmp/{port}/{name}",
        data_dir=data_dir,
        clusters=config,
        port=port,
        log_level="debug",
        cache_size=cache_size,
        force_create=True,
        request_ttl=timeout)

    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


# TODO: delete
class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 4
    NAME = "persistence"

    def cluster(self, chain_type):
        raise NotImplementedError

    def execute(self, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.chain.Block)

    def testSyncChain(self):
        self._execute(tc.chain.Sync)

    def _execute(self, chain_type):
        name = self.NAME

        cluster = self.cluster(chain_type)

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.uri(cluster).path()
            host = start_host(f"test_{name}_{i}", [cluster], host_uri=tc.URI(host_uri), cache_size=self.CACHE_SIZE)
            hosts.append(host)
            printlines(5)

        time.sleep(1)

        self.execute(hosts)

        for host in hosts:
            host.stop()


def maybe_create_dir(path, force):
    path = pathlib.Path(path)
    if path.exists() and path.is_dir():
        return
    elif force:
        os.makedirs(path)
    else:
        raise RuntimeError(f"no directory at {path}")


def printlines(n):
    for _ in range(n):
        print()
