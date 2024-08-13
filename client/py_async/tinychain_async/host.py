import aiohttp
import asyncio
import inspect
import json

from tinychain.context import to_json
from tinychain.error import BadRequest, Conflict, Forbidden, MethodNotAllowed, NotFound, Timeout, Unauthorized, UnknownError
from tinychain.service import Library, Model, Service
from tinychain.uri import URI

from tinychain.host import Local, Host


BACKOFF = 0.01
TIMEOUT = aiohttp.ClientTimeout(total=86400)


class Host(Host):
    RETRYABLE = [409, 502, 503]

    async def _handle(self, response):
        status = response.status

        try:
            response = await response.json()
        except json.decoder.JSONDecodeError as cause:
            response = await response.text()
            raise ValueError(f"invalid JSON response: {response} ({cause}")

        if status == 200:
            return response
        elif status == 204:
            return None

        if status == BadRequest.CODE:
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

    async def get(self, path, key=None):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.get(endpoint, params={"key": json.dumps(to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(to_json(key))})
                backoff *= 2

            return response

    async def put(self, path, key=None, value=None):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.put(endpoint, params={"key": json.dumps(to_json(key))}, json=to_json(value))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.put(endpoint, params={"key": json.dumps(to_json(key))}, json=value)
                backoff *= 2

            return response

    async def post(self, path, params={}):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.post(endpoint, json=to_json(params))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.post(endpoint, json=to_json(params))
                backoff *= 2

            return response

    async def delete(self, path, key=None):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.delete(endpoint, params={"key": json.dumps(to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(to_json(key))})
                backoff *= 2

            return response

    async def create_namespace(self, path):
        """Create a directory at the given `path`."""

        exists = 1
        while exists < len(path):
            try:
                await self._handle(await self.get(path[:exists + 1]))
                exists += 1
            except NotFound:
                break

        for i in range(exists, len(path) - 1):
            await self._handle(await self.put(path[:i], path[i], True))

        return await self.put(path[:-1], path[-1], False)

    async def hypothetical(self, op_def):
        """Execute the given `op_def` without committing any writes."""

        return await self.post("/transact/hypothetical", {"op": op_def})

    async def install(self, lib_or_service):
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
            class_path = URI(Model) + install_path[1:]
            await self.create_namespace(class_path[:-1])
            await self._handle(await self.put(class_path[:-1], class_path[-1], class_set))

        await self.create_namespace(install_path[:-1])
        return await self._handle(await self.put(install_path[:-1], install_path[-1], lib_or_service))

    async def update(self, lib_or_service):
        """Update the version of given `lib_or_service` on this host"""

        install_path = URI(lib_or_service).path()
        return await self.put(install_path[:-1], install_path[-1], lib_or_service)


class Local(Host, Local):
    pass
