"""Utilities for communicating with a TinyChain host."""
import abc
import json
import requests
import urllib.parse

from tinychain.error import *
from tinychain.util import to_json, uri, URI
from tinychain.value import Nil


DEFAULT_PORT = 8702
ENCODING = "utf-8"


class Host(object):
    """A TinyChain host."""

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

        try:
            response = json.loads(response.text)
        except json.decoder.JSONDecodeError as cause:
            raise ValueError(f"invalid JSON response: {response.text} ({cause}")

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
        elif status == 409:
            raise Conflict(response)
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
        if key and not isinstance(key, Nil):
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.get(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.get(url, headers=headers)

        return self._handle(request)

    def put(self, path, key=None, value=None, auth=None):
        """Execute a PUT request."""

        url = self.link(path)
        headers = auth_header(auth)
        if key and not isinstance(key, Nil):
            key = json.dumps(to_json(key)).encode(ENCODING)
            params = {"key": key}
        else:
            params = {}

        value = json.dumps(to_json(value)).encode(ENCODING)
        request = lambda: requests.put(url, params=params, data=value, headers=headers)

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
        if key and not isinstance(key, Nil):
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.delete(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.delete(url, headers=headers)

        return self._handle(request)


class Local(Host):
    """A local TinyChain host."""

    class Process(object):
        """A local TinyChain host process."""

        @abc.abstractmethod
        def start(self, wait_time):
            """Start this host `Process`."""

        @abc.abstractmethod
        def stop(self, wait_time):
            """Shut down this host `Process`."""

        def __del__(self):
            if self._process:
                self.stop()

    SHUTDOWN_TIME = 0.1
    STARTUP_TIME = 1.

    def __init__(self, process, address):
        self._process = process
        Host.__init__(self, address)

    def start(self, wait_time=STARTUP_TIME):
        """Start this local `Host`."""

        self._process.start(wait_time)

    def stop(self, wait_time=SHUTDOWN_TIME):
        """Shut down this local `Host`."""

        self._process.stop(wait_time)

    def __del__(self):
        self._process.stop(self.SHUTDOWN_TIME)


def auth_header(token):
    return {"Authorization": f"Bearer {token}"} if token else {}
