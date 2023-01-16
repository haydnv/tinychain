import aiohttp
import asyncio
import json

from ..context import to_json
from ..uri import URI

from .sync import auth_header, Local, Host


BACKOFF = 0.01
TIMEOUT = aiohttp.ClientTimeout(total=86400)


class Host(Host):
    RETRYABLE = [409, 502, 503]

    async def get(self, path, key=None):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.get(endpoint, params={"key": json.dumps(to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(key)})
                backoff *= 2

            return response

    async def put(self, path, key=None, value=None, auth=None):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(headers=auth_header(auth), timeout=TIMEOUT) as session:
            response = await session.put(endpoint, params={"key": json.dumps(to_json(key))}, json=to_json(value))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.put(endpoint, params={"key": json.dumps(key)}, json=value)
                backoff *= 2

            return response

    async def post(self, path, params={}):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.post(endpoint, json=to_json(params))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.post(endpoint, json=params)
                backoff *= 2

            return response

    async def delete(self, path, key=None):
        endpoint = str(URI(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.delete(endpoint, params={"key": json.dumps(to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(key)})
                backoff *= 2

            return response

    async def create_namespace(self, actor, base_dir, ns, lead=None):
        return await self.put(*self._namespace_args(actor, base_dir, ns, lead))

    async def install(self, actor, service):
        return await self.put(*self._install_args(actor, service))


class Local(Host, Local):
    pass
