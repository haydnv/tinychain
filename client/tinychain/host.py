"""Utilities for communicating with a TinyChain host."""

import abc
import inspect
import logging

import json
import requests
import urllib.parse

from .app import Library
from .context import to_json
from .error import *
from .scalar.value import Nil
from .state import State
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

    def install(self, library):
        """Install the given `library` on this host"""

        if not URI(library).path().startswith(str(URI(Library))):
            raise ValueError(f"not a library: {library}")

        deps = []
        classes = {}
        for name, cls in inspect.getmembers(library, inspect.isclass):
            if issubclass(cls, Library):
                deps.append(cls)
            elif issubclass(cls, State):
                if cls.NS == library.NS:
                    if URI(cls) == (URI("/class") + library.NS).append(library.NAME).append(cls.__name__):
                        classes[cls.__name__] = cls
                    else:
                        raise ValueError(f"{library} class {cls} has invalid URI {URI(cls)}")
                else:
                    raise ValueError(
                        f"{library} may not depend on {cls} directly but must depend on its Library in {cls.NS}")
            else:
                logging.info(f"install {library} will skip dependency {cls} since it's not a Library or a State")

        if deps:
            logging.info(f"installing {library} which depends on {deps}...")

        # TODO: don't hard-code URI("/class")
        if classes:
            self.put(URI("/class") + library.NS, library.NAME, classes)

        self.put(URI(library).path()[:-2], library.NAME, library)

    def update(self, library):
        """Update the version of given `library` on this host"""

        self.put(library.URI.path(), library.VERSION, library)


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
