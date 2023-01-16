import docker
import logging
import os
import pathlib
import shutil
import subprocess
import time

import rjwt
import tinychain as tc


CONFIG = "config"
DEFAULT_PORT = 8702
DEFAULT_WORKSPACE = "/tmp/tc/tmp"
DOCKERFILE = os.getenv("TC_DOCKER", "tests/tctest/Dockerfile")
DOCKER_NETWORK_MODE = os.getenv("DOCKER_MODE", "host")
TC_PATH = os.getenv("TC_PATH", "host/target/debug/tinychain")


class Docker(tc.host.Local.Process):
    ADDRESS = "127.0.0.1"

    def __init__(self, **flags):
        if "data_dir" in flags:
            raise RuntimeError(f"the test Dockerfile does not support a custom data directory path {flags['data_dir']}")

        if "log_level" not in flags:
            flags["log_level"] = "debug"

        self.client = docker.from_env()
        self.container = None
        self._flags = flags

    def start(self, wait_time):
        """Build and run a Docker image from the local repository."""

        if self.container:
            raise RuntimeError("tried to start a Docker container that's already running")

        # build the Docker image
        print("building Docker image")
        try:
            (image, _logs) = self.client.images.build(path=".", dockerfile=DOCKERFILE)
        except Exception as e:
            exit(f"building Docker image failed: {e}")

        print("built Docker image")

        # construct the TinyChain host arguments
        cmd = ["/tinychain", "--data_dir=/data"]
        for flag, value in self._flags.items():
            cmd.append(f"--{flag}={value}")
        cmd = ' '.join(cmd)

        # run a Docker container
        print(f"running new Docker container with image {image.id}...")
        self.container = self.client.containers.run(
            image.id, cmd,
            network_mode=DOCKER_NETWORK_MODE,
            detach=True)

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


def start_docker(ns, host_uri=None, public_key=None, wait_time=1., **flags):
    assert ns.startswith('/'), f"namespace must be a URI path, not {ns}"

    if not os.path.exists(DOCKERFILE):
        raise RuntimeError(
            f"Dockerfile at {DOCKERFILE} not found--use the TC_DOCKER environment variable to set a different path")

    port = DEFAULT_PORT
    if flags.get("http_port"):
        port = flags["http_port"]
        del flags["http_port"]
    elif host_uri is not None and host_uri.port():
        port = host_uri.port()

    if public_key:
        flags["public_key"] = public_key.hex()

    process = Docker(http_port=port, **flags)
    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


class Local(tc.host.Local.Process):
    """A local TinyChain host process."""

    ADDRESS = "127.0.0.1"

    def __init__(self, path, workspace, force_create, **flags):
        http_port = flags.get("http_port", DEFAULT_PORT)
        print(f"start host process on port {http_port}")

        # set _process first so it's available to __del__ in case of an exception
        self._process = None

        if http_port is None or not int(http_port) or int(http_port) < 0:
            raise ValueError(f"invalid port: {http_port}")

        if "data_dir" in flags:
            _maybe_create_dir(flags["data_dir"])

        args = [
            path,
            f"--workspace={workspace}",
            f"--address={self.ADDRESS}",
        ]

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

        logging.info(f"shutting down TinyChain host")
        if self._process:
            self._process.terminate()
            self._process.wait()
            logging.info(f"host shut down")

            if wait_time:
                time.sleep(wait_time)
        else:
            logging.info(f"host not running")

        self._process = None

    def __del__(self):
        if self._process:
            self.stop()


def _start_local_host_process(ns, host_uri=None, public_key=None, wait_time=1, **flags):
    assert ns.startswith('/'), f"namespace must be a URI path, not {ns}"
    name = str(ns)[1:].replace('/', '_')

    if not os.path.isfile(TC_PATH):
        hint = "use the TC_PATH environment variable to set the path to the TinyChain host binary"
        raise RuntimeError(f"invalid executable path: {TC_PATH} ({hint})")

    port = DEFAULT_PORT
    if flags.get("http_port"):
        port = flags["http_port"]
        del flags["http_port"]
    elif host_uri is not None and host_uri.port():
        port = host_uri.port()

    if public_key:
        flags["public_key"] = public_key.hex()

    data_dir = f"/tmp/tc/data/{port}/{name}"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    if "log_level" not in flags:
        flags["log_level"] = "debug"

    if "workspace" in flags:
        workspace = flags["workspace"] + f"/{port}/{name}"
        del flags["workspace"]
    else:
        workspace = DEFAULT_WORKSPACE + f"/{port}/{name}"

    process = Local(
        TC_PATH,
        workspace=workspace,
        force_create=True,
        data_dir=data_dir,
        http_port=port,
        **flags)

    process.start(wait_time)

    return process, port

def start_local_host(ns, host_uri=None, public_key=None, wait_time=1, **flags):
    process, port = _start_local_host_process(ns, host_uri, public_key, wait_time, **flags)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


def start_local_host_async(ns, host_uri=None, public_key=None, wait_time=1, **flags):
    process, port = _start_local_host_process(ns, host_uri, public_key, wait_time, **flags)
    return tc.host.asynchronous.Local(process, f"http://{process.ADDRESS}:{port}")


# use this alias to switch between Local and Docker host types
if DOCKERFILE == "0":
    start_host = start_local_host
else:
    start_host = start_docker


def _maybe_create_dir(path):
    path = pathlib.Path(path)
    if path.exists() and path.is_dir():
        return
    else:
        os.makedirs(path)
