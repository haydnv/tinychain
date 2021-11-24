import docker
import logging
import time
import tinychain as tc

DEFAULT_PORT = 8702


class Process(tc.host.Local.Process):
    ADDRESS = "127.0.0.1"
    BUILD = "docker build -f tests/client/Dockerfile ."

    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def start(self, wait_time):
        """Build and run a Docker image from the local repository."""

        if self.container:
            raise RuntimeError("tried to start a Docker container that's already running")

        # build the Docker image
        with open("Dockerfile", 'rb') as dockerfile:
            print("building Docker image")
            (image, _logs) = self.client.images.build(path=".", fileobj=dockerfile)
            print("built Docker image")

        # run a Docker container
        print("running new Docker container")
        self.container = self.client.containers.run(image.id, "/tinychain", network_mode="host", detach=True)
        time.sleep(wait_time)
        print("new Docker container running")

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


def start_host(host_uri=None, wait_time=1.):
    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()

    process = Process()
    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")
