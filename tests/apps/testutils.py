import docker
import logging
import os
import pathlib
import shutil
import subprocess
import time
import tinychain as tc

CONFIG = "config"
DEFAULT_PORT = 8702
DOCKER_NETWORK_MODE = os.getenv("DOCKER_MODE", "host")
TC_PATH = os.getenv("TC_PATH", "host/target/debug/tinychain")


class Docker(tc.host.Local.Process):
    ADDRESS = "127.0.0.1"
    BUILD = "docker build -f tests/apps/Dockerfile ."

    def __init__(self, config_dir, clusters, **flags):
        self.client = docker.from_env()
        self.clusters = clusters
        self.config_dir = config_dir
        self.container = None
        self._flags = flags

    def start(self, wait_time):
        """Build and run a Docker image from the local repository."""

        if self.container:
            raise RuntimeError("tried to start a Docker container that's already running")

        # build the Docker image
        print("building Docker image")
        (image, _logs) = self.client.images.build(path=".", dockerfile="tests/apps/Dockerfile")
        print("built Docker image")

        # construct the TinyChain host arguments
        cmd = ["/tinychain", "--data_dir=/data"]
        cmd.extend(f"--cluster=/{CONFIG}{cluster}" for cluster in self.clusters)
        for flag, value in self._flags.items():
            cmd.append(f"--{flag}={value}")
        cmd = ' '.join(cmd)

        # run a Docker container
        print(f"running new Docker container with image {image.id}")
        self.container = self.client.containers.run(
            image.id, cmd,
            network_mode=DOCKER_NETWORK_MODE, volumes=[f"{self.config_dir}:/{CONFIG}"], detach=True)

        time.sleep(wait_time)
        print(self.container.logs().decode("utf-8"))

    def stop(self, _wait_time=None):
        """Stop this Docker container."""

        if self.container:
            print("stopping Docker container")
            self.container.stop()
            print("Docker container stopped")
            print()
        else:
            logging.info(f"Docker container not running")

        self.container = None

    def __del__(self):
        self.stop()


def start_docker(name, app_or_library, overwrite=True, host_uri=None, wait_time=1., **flags):
    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()

    config_dir = os.getcwd()
    config_dir += f"/{CONFIG}/{name}/{port}"
    maybe_create_dir(config_dir, overwrite)

    app_configs = []

    deps = tc.app.dependencies(app_or_library) if isinstance(app_or_library, tc.app.Library) else app_or_library

    for lib in deps:
        lib_path = tc.uri(lib).path()
        tc.app.write_config(lib, f"{config_dir}{lib_path}", overwrite)
        app_configs.append(lib_path)

    process = Docker(config_dir, app_configs, http_port=port, **flags)
    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


class Local(tc.host.Local.Process):
    """A local TinyChain host process."""

    ADDRESS = "127.0.0.1"

    def __init__(self, path, workspace, force_create, libs, **flags):
        http_port = flags.get("http_port", DEFAULT_PORT)
        print(f"start host process on port {http_port}")

        # set _process first so it's available to __del__ in case of an exception
        self._process = None

        if http_port is None or not int(http_port) or int(http_port) < 0:
            raise ValueError(f"invalid port: {http_port}")

        if libs and "data_dir" not in flags:
            raise ValueError("hosting a cluster requires specifying a data_dir")

        if "data_dir" in flags:
            maybe_create_dir(flags["data_dir"], force_create)

        args = [
            path,
            f"--workspace={workspace}",
            f"--address={self.ADDRESS}",
        ]

        args.extend(f"--cluster={dep}" for dep in libs)
        args.extend(f"--{flag}={value}" for flag, value in flags.items())

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


def start_local_host(name, app_or_library, overwrite=True, host_uri=None, cache_size="5K", wait_time=1, **flags):
    if not os.path.isfile(TC_PATH):
        raise RuntimeError(f"invalid executable path: {TC_PATH}")

    deps = tc.app.dependencies(app_or_library) if isinstance(app_or_library, tc.app.Library) else app_or_library

    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()
    elif deps and tc.uri(deps[0]).port():
        port = tc.uri(deps[0]).port()

    config_dir = os.getcwd()
    config_dir += f"/{CONFIG}/{name}/{port}"
    maybe_create_dir(config_dir, overwrite)

    app_configs = []
    for dep in deps:
        app_path = tc.uri(dep).path()
        app_path = f"{config_dir}{app_path}"
        tc.app.write_config(dep, app_path, overwrite)
        app_configs.append(app_path)

    data_dir = f"/tmp/tc/tmp/{port}/{name}"
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    process = Local(
        TC_PATH,
        workspace=f"/tmp/tc/tmp/{port}/{name}",
        libs=app_configs,
        force_create=True,
        data_dir=data_dir,
        http_port=port,
        log_level="debug",
        cache_size=cache_size,
        **flags)

    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


# use this alias to switch between Local and Docker host types
start_host = start_docker


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
