"""Utilities for communicating with a TinyChain host."""

import abc
import inspect
import logging

import json
import requests
import rjwt
import urllib.parse

from ..service import Library, Model, Service
from ..context import to_json
from ..error import *
from ..scalar.value import Nil
from ..state import State
from ..uri import URI


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

    def _namespace_args(self, actor, base_dir, ns, lead):
        ns = URI(ns)
        lead = None if lead is None else URI(lead)

        if ns.host() is not None:
            raise ValueError(f"namespace {ns} should not include a host")

        if len(ns) != 1:
            raise NotImplementedError(f"create a namespace {ns} (len {len(ns)}) by creating directories recursively")

        if lead is not None and lead.path() is not None:
            raise ValueError(f"lead for {ns} ({lead}) should not have a path component")

        logging.info(f"create namespace {base_dir}{ns} on {self}")

        auth_host = lead if lead else URI(self)

        issuer = str(auth_host + base_dir)
        token = rjwt.Token.issue(issuer, '/', [str(ns)])
        token = actor.sign_token(token)

        issuer = str(auth_host + URI(Model))
        token = rjwt.Token.consume(token, issuer, '/', [str(ns)])
        token = actor.sign_token(token)

        lead = base_dir + ns if lead is None else lead + base_dir + ns
        return base_dir, str(ns)[1:], lead, token

    def create_namespace(self, actor, base_dir, ns, lead=None):
        return self.put(*self._namespace_args(actor, base_dir, ns, lead))

    def _install_args(self, actor, service):
        if isinstance(service, Service):
            if not URI(service).path().startswith(URI(Service)):
                raise ValueError(f"invalid path for Service: {URI(service)}")
        elif isinstance(service, Library):
            if not URI(service).path().startswith(URI(Library)):
                raise ValueError(f"invalid path for Library: {URI(service)}")
        else:
            raise ValueError(f"not a Library or Service: {service}")

        lead = URI(service)[0] if URI(service).host() else None
        path = URI(service).path()
        base_dir = path[0]
        ns = path[1:-2]
        name = path[-2]
        _version = path[-1]

        logging.info(f"install {name} at {base_dir}{ns} on {self}")

        if URI(service).host() is None:
            lead = URI(self)
        else:
            lead = URI(service)[0]

        deps = []
        for _, cls in inspect.getmembers(service, inspect.isclass):
            if issubclass(cls, Library):
                deps.append(cls)
            elif issubclass(cls, State):
                continue
            else:
                logging.info(f"install {service} will skip dependency {cls} since it's not a Library or a State")

        if deps:
            logging.info(f"installing {service} which depends on {deps}...")
        else:
            logging.info(f"installing {service}")

        auth_host = URI(lead) if lead else URI(self)
        issuer = str(auth_host + base_dir + ns)
        token = rjwt.Token.issue(issuer, '/', [str(name)])
        token = actor.sign_token(token)

        issuer = str(auth_host + URI(Model) + ns)
        token = rjwt.Token.consume(token, issuer, '/', [str(name)])
        token = actor.sign_token(token)

        return URI(service).path()[:-2], str(name)[1:], service, token

    def install(self, actor, service):
        """Install the given `service` on this host"""

        return self.put(*self._install_args(actor, service))

    def update(self, actor, service):
        """Update the version of given `service` on this host"""

        lead = URI(service)[0] if URI(service).host() else None
        path = URI(service).path()
        base_dir = path[0]
        ns = path[1:-2]
        name = path[-2]
        version = path[-1]

        auth_host = URI(lead) if lead else URI(self)
        issuer = str(auth_host + base_dir + ns)
        token = rjwt.Token.issue(issuer, '/', [str(name + version)])
        token = actor.sign_token(token)

        issuer = str(auth_host + URI(Model) + ns)
        token = rjwt.Token.consume(token, issuer, '/', [str(name)])
        token = actor.sign_token(token)

        return self.put(URI(service).path()[:-1], str(version)[1:], service, token)


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
