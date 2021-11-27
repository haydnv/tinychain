import docker
import logging
import os
import pathlib
import time
import tinychain as tc

CONFIG = "config"
DEFAULT_PORT = 8702


class Process(tc.host.Local.Process):
    ADDRESS = "127.0.0.1"
    BUILD = "docker build -f tests/client/Dockerfile ."

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


def start_host(name, clusters, overwrite=True, host_uri=None, wait_time=1.):
    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()

    config_dir = os.getcwd()
    config_dir += f"/{CONFIG}/{name}/{port}"
    maybe_create_dir(config_dir, overwrite)

    cluster_configs = []
    for cluster in clusters:
        cluster_path = tc.uri(cluster).path()
        tc.write_cluster(cluster, f"{config_dir}{cluster_path}", overwrite)
        cluster_configs.append(cluster_path)

    process = Process(config_dir, cluster_configs)
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
