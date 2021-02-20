import json
import os
import pathlib
import requests
import subprocess
import sys
import time
import urllib.parse

from .error import *
from .util import to_json, uri


ENCODING = "utf-8"


class Host(object):
    @staticmethod
    def encode_params(params):
        return urllib.parse.urlencode({k: json.dumps(v) for k, v in params.items()})

    def __init__(self, address):
        self.address = address

    def _handle(self, req):
        response = req()
        status = response.status_code
        response = response.text

        if status == 200:
            return json.loads(response)
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
        elif status == 501:
            raise NotImplemented(response)
        else:
            raise UnknownError(f"HTTP error code {status}: {response}")

    def link(self, path):
        return "http://{}{}".format(self.address, path)

    def get(self, path, key=None, auth=None):
        url = self.link(path)
        headers = auth_header(auth)
        if key:
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.get(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.get(url, headers=headers)

        return self._handle(request)

    def put(self, path, key, value, auth=None):
        url = self.link(path)
        headers = auth_header(auth)
        key = json.dumps(to_json(key)).encode(ENCODING)
        value = json.dumps(to_json(value)).encode(ENCODING)
        request = lambda: requests.put(url, params={"key": key}, data=value, headers=headers)

        return self._handle(request)

    def post(self, path, data, auth=None):
        url = self.link(path)
        data = json.dumps(to_json(data)).encode(ENCODING)
        headers = auth_header(auth)
        request = lambda: requests.post(url, data=data, headers=headers)

        return self._handle(request)

    def delete(self, path, key=None, auth=None):
        url = self.link(path)
        headers = auth_header(auth)
        if key:
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.delete(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.delete(url, headers=headers)

        return self._handle(request)

    def resolve(self, state, auth=None):
        return self.post("/transact/execute", state, auth)


class Local(Host):
    ADDRESS = "127.0.0.1"
    STARTUP_TIME = 1.0

    def __init__(self, workspace, data_dir=None, clusters=[], force_create=False):
        Host.__init__(self, self.ADDRESS)
        self._process = None

        if clusters and data_dir is None:
            raise ValueError("Hosting a cluster requires specifying a data_dir")

        maybe_create_dir(workspace, force_create)

        if data_dir:
            maybe_create_dir(data_dir, force_create)

        self.clusters = clusters
        self.data_dir = data_dir
        self.workspace = workspace

    def start(self, path, port, log_level="warn"):
        if not int(port) or int(port) < 0:
            raise ValueError(f"invalid port: {port}")

        address = "{}:{}".format(__class__.ADDRESS, port)
        Host.__init__(self, address)

        args = [
            path,
            f"--http_port={port}",
            f"--log_level={log_level}",
        ]

        if self.data_dir:
            args.append(f"--data_dir={self.data_dir}")

        args.extend([f"--cluster={cluster}" for cluster in self.clusters])

        self._process = subprocess.Popen(args)
        time.sleep(__class__.STARTUP_TIME)

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(f"Tinychain process at {self.address} crashed on startup")
        else:
            print(f"new instance running at {self.address}")

    def terminate(self):
        if self._process.poll() != None:
            return

        print(f"Terminating Tinychain host {self.address}")
        self._process.terminate()
        self._process.wait()
        print(f"Host {self.address} Terminated")

    def __del__(self):
        if self._process:
            self.terminate()


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

