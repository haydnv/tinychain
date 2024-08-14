"""Utilities for communicating with a TinyChain host."""

import abc
import inspect

import json

import requests
import urllib.parse

from .service import Library, Model, Service
from .context import to_json
from .error import BadRequest, Conflict, Forbidden, MethodNotAllowed, NotFound, NotImplemented, Timeout, Unauthorized, UnknownError
from .scalar.value import Nil
from .uri import URI


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

        if URI(self).path():
            raise ValueError(f"Host address should not include the application path {URI(self).path()}")

    def __repr__(self):
        return f"host at {URI(self)}"

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
        elif status == BadRequest.CODE:
            raise BadRequest(response)
        elif status == Unauthorized.CODE:
            raise Unauthorized(response)
        elif status == Forbidden.CODE:
            raise Forbidden(response)
        elif status == NotFound.CODE:
            raise NotFound(response)
        elif status == MethodNotAllowed.CODE:
            raise MethodNotAllowed(response)
        elif status == Timeout.CODE:
            raise Timeout(response)
        elif status == Conflict.CODE:
            raise Conflict(response)
        elif status == NotImplemented.CODE:
            raise NotImplemented(response)
        else:
            raise UnknownError(f"HTTP error code {status}: {response}")

    def link(self, path):
        """Return a link to the given path at this host."""

        if hasattr(path, "__uri__"):
            path = URI(path)
        else:
            path = path if isinstance(path, URI) else URI(path)

        return URI(self) + path

    def get(self, path, key=None, auth=None):
        """Execute a GET request."""

        if isinstance(path, URI) and path.host() is not None:
            raise ValueError(f"Host.get expects a path, not {path}")

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

        if isinstance(path, URI) and path.host() is not None:
            raise ValueError(f"Host.put expects a path, not {path}")

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

        if isinstance(path, URI) and path.host() is not None:
            raise ValueError(f"Host.post expects a path, not {path}")

        url = self.link(path)
        data = json.dumps(to_json(data)).encode(ENCODING)
        headers = auth_header(auth)
        request = lambda: requests.post(url, data=data, headers=headers)

        return self._handle(request)

    def delete(self, path, key=None, auth=None):
        """Execute a DELETE request."""

        if isinstance(path, URI) and path.host() is not None:
            raise ValueError(f"Host.delete expects a path, not {path}")

        url = self.link(path)
        headers = auth_header(auth)
        if key and not isinstance(key, Nil):
            key = json.dumps(to_json(key)).encode(ENCODING)
            request = lambda: requests.delete(url, params={"key": key}, headers=headers)
        else:
            request = lambda: requests.delete(url, headers=headers)

        return self._handle(request)

    def create_namespace(self, path):
        """Create a directory at the given `path`."""

        exists = 1
        while exists < len(path):
            try:
                self.get(path[:exists + 1])
                exists += 1
            except NotFound:
                break

        for i in range(exists, len(path) - 1):
            self.put(path[:i], path[i], True)

        self.put(path[:-1], path[-1], False)

    def hypothetical(self, op_def, auth=None):
        """Execute the given `op_def` without committing any writes."""

        return self.post("/transact/hypothetical", {"op": op_def}, auth)

    def install(self, lib_or_service):
        """Install the given `lib_or_service` on this host"""

        install_path = URI(lib_or_service).path()
        assert install_path
        assert install_path[:1] in [URI(Library), URI(Service)]

        # TODO: replace this with a package manager
        class_set = {}
        lib_or_service_json = to_json(lib_or_service)[URI(lib_or_service)[:-1]]
        for name, attr in inspect.getmembers(lib_or_service, inspect.isclass):
            if name in lib_or_service_json:
                class_set[name] = lib_or_service_json[name]

        if class_set:
            class_uri = URI(Model) + install_path[1:]
            self.create_namespace(class_uri[:-1])
            self.put(class_uri[:-1], class_uri[-1], class_set)

        self.create_namespace(install_path[:-1])
        self.put(install_path[:-1], install_path[-1], lib_or_service)

    def update(self, lib_or_service):
        """Update the version of given `lib_or_service` on this host"""

        install_path = URI(lib_or_service).path()
        return self.put(install_path[:-1], install_path[-1], lib_or_service)


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
