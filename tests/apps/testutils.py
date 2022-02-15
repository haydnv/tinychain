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
TC_PATH = os.getenv("TC_PATH", "host/target/debug/tinychain")


class Docker(tc.host.Local.Process):
    ADDRESS = "127.0.0.1"
    BUILD = "docker build -f tests/apps/Dockerfile ."

    def __init__(self, config_dir, clusters):
        self.client = docker.from_env()
        self.clusters = clusters
        self.config_dir = config_dir
        self.container = None

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
        cmd = ' '.join(cmd)

        # run a Docker container
        print(f"running new Docker container with image {image.id}")
        self.container = self.client.containers.run(
            image.id, cmd,
            network_mode="host", volumes=[f"{self.config_dir}:/{CONFIG}"], detach=True)

        time.sleep(wait_time)
        print(self.container.logs().decode("utf-8"))

    def stop(self, _wait_time=None):
        """Stop this Docker container."""

        if self.container:
            print("stopping Docker container")
            self.container.stop()
            print("Docker container stopped")
        else:
            logging.info(f"Docker container not running")

        self.container = None

    def __del__(self):
        self.stop()


def start_docker(name, apps, overwrite=True, host_uri=None, wait_time=1.):
    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()

    config_dir = os.getcwd()
    config_dir += f"/{CONFIG}/{name}/{port}"
    maybe_create_dir(config_dir, overwrite)

    app_configs = []
    for app in apps:
        app_path = tc.uri(app).path()
        tc.app.write_config(app, f"{config_dir}{app_path}", overwrite)
        app_configs.append(app_path)

    process = Docker(config_dir, app_configs)
    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


class Local(tc.host.Local.Process):
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


def start_host(name, libs=[], overwrite=True, host_uri=None, cache_size="5K", wait_time=1, timeout=30):
    if not os.path.isfile(TC_PATH):
        raise RuntimeError(f"invalid executable path: {TC_PATH}")

    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()
    elif libs and tc.uri(libs[0]).port():
        port = tc.uri(libs[0]).port()

    config_dir = os.getcwd()
    config_dir += f"{CONFIG}/{name}/{port}"
    maybe_create_dir(config_dir, overwrite)

    app_configs = []
    for lib in libs:
        app_path = tc.uri(lib).path()
        app_path = f"{config_dir}{app_path}"
        tc.app.write_config(lib, app_path, overwrite)
        app_configs.append(app_path)

    data_dir = f"/tmp/tc/tmp/{port}/{name}"
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    process = Local(
        TC_PATH,
        workspace=f"/tmp/tc/tmp/{port}/{name}",
        data_dir=data_dir,
        clusters=app_configs,
        port=port,
        log_level="debug",
        cache_size=cache_size,
        force_create=True,
        request_ttl=timeout)

    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


def maybe_create_dir(path, force):
    path = pathlib.Path(path)
    if path.exists() and path.is_dir():
        return
    elif force:
        os.makedirs(path)
    else:
        raise RuntimeError(f"no directory at {path}")
