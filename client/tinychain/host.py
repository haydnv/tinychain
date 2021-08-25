"""Utilities for communicating with a Tinychain host."""

import json
import logging
import os
import pathlib
import requests
import subprocess
import time
import urllib.parse

from tinychain.error import *
from tinychain.util import to_json, uri, URI


DEFAULT_PORT = 8702
ENCODING = "utf-8"


class Host(object):
    """A Tinychain host."""

    @staticmethod
    def encode_params(params):
        return urllib.parse.urlencode({k: json.dumps(v) for k, v in params.items()})

    def __init__(self, address):
        if "://" not in address:
            raise ValueError(f"host address missing protocol: {address}")

        self.__uri__ = URI(address)

        if uri(self).path():
            raise ValueError(f"Host address should not include the application path {uri(self).path()}")

    def _handle(self, req):
        response = req()
        status = response.status_code
        response = json.loads(response.text)

        if status == 200:
            return response
        elif status == 204:
            return None
        elif status == 400:
            raise BadRequest(response)
        elif status == 401:
            raise Unauthorized(response)
        elif status == 403:
            raise Forbidden(response)
        elif status == 404:
            raise NotFound(response)
        elif status == 405:
            raise MethodNotAllowed(response)
        elif status == 408:
            raise Timeout(response)
        elif status == 501:
            raise NotImplemented(response)
        else:
            raise UnknownError(f"HTTP error code {status}: {response}")

    def link(self, path):
        """Return a link to the given path at this host."""

        return uri(self) + path

    def get(self, path, key=None, auth=None):
        """Execute a GET request."""

        url = self.link(path)
        headers = auth_header(auth)
        if key:
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.get(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.get(url, headers=headers)

        return self._handle(request)

    def put(self, path, key=None, value=None, auth=None):
        """Execute a PUT request."""

        url = self.link(path)
        headers = auth_header(auth)
        key = json.dumps(to_json(key)).encode(ENCODING)
        value = json.dumps(to_json(value)).encode(ENCODING)
        request = lambda: requests.put(url, params={"key": key}, data=value, headers=headers)

        return self._handle(request)

    def post(self, path, data={}, auth=None):
        """Execute a POST request."""

        url = self.link(path)
        data = json.dumps(to_json(data)).encode(ENCODING)
        headers = auth_header(auth)
        request = lambda: requests.post(url, data=data, headers=headers)

        return self._handle(request)

    def delete(self, path, key=None, auth=None):
        """Execute a DELETE request."""

        url = self.link(path)
        headers = auth_header(auth)
        if key:
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.delete(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.delete(url, headers=headers)

        return self._handle(request)


class Local(Host):
    """A local Tinychain host."""

    ADDRESS = "127.0.0.1"
    SHUTDOWN_TIME = 0.1
    STARTUP_TIME = 1.

    def __init__(self,
            path,
            workspace,
            data_dir=None,
            clusters=[],
            port=DEFAULT_PORT,
            log_level="warn",
            cache_size="1G",
            force_create=False):

        # set _process first so it's available to __del__ in case of an exception
        self._process = None

        if port is None or not int(port) or int(port) < 0:
            raise ValueError(f"invalid port: {port}")

        if clusters and data_dir is None:
            raise ValueError("Hosting a cluster requires specifying a data_dir")

        maybe_create_dir(workspace, force_create)

        if data_dir:
            maybe_create_dir(data_dir, force_create)

        address = "http://{}:{}".format(self.ADDRESS, port)
        Host.__init__(self, address)

        args = [
            path,
            f"--workspace={workspace}",
            f"--http_port={port}",
            f"--log_level={log_level}",
            f"--cache_size={cache_size}",
        ]

        if data_dir:
            args.append(f"--data_dir={data_dir}")

        args.extend([f"--cluster={cluster}" for cluster in clusters])

        self._args = args

    def start(self, wait_time=STARTUP_TIME):
        """Start this host process locally."""

        if self._process:
            raise RuntimeError("tried to start a host that's already running")

        self._process = subprocess.Popen(self._args)        
        time.sleep(wait_time)

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(f"Tinychain process at {uri(self)} crashed on startup")
        else:
            logging.info(f"new instance running at {uri(self)}")

    def stop(self, wait_time=SHUTDOWN_TIME):
        """Shut down this host."""

        logging.info(f"Shutting down Tinychain host {uri(self)}")
        if self._process:
            self._process.terminate()
            self._process.wait()
            logging.info(f"Host {uri(self)} shut down")
            time.sleep(wait_time)
        else:
            logging.info(f"{uri(self)} not running")

        self._process = None

    def wait(self):
        """Block the current thread until the running host is shut down."""
        self._process.wait()

    def __del__(self):
        if self._process:
            self.stop()


def auth_header(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def maybe_create_dir(path, force):
    path = pathlib.Path(path)
    if path.exists() and path.is_dir():
        return
    elif force:
        os.makedirs(path)
    else:
        raise RuntimeError(f"no directory at {path}")

