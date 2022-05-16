import aiohttp
import argparse
import asyncio
import inspect
import json
import logging
import random
import time
import tinychain as tc

from .process import start_local_host

DEFAULT_CONCURRENCY = 5
BACKOFF = 0.01
TIMEOUT = aiohttp.ClientTimeout(total=86400)


# TODO: move this class into a new async module in tc.host
class Host(object):
    RETRYABLE = [409, 502, 503]

    def __init__(self, host):
        self.__uri__ = tc.uri(host)
        self._host = host  # keep a reference here so it doesn't get dropped

    async def get(self, path, key=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.get(endpoint, params={"key": json.dumps(tc.to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(key)})
                backoff *= 2

            return response

    async def put(self, path, key=None, value=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.put(endpoint, params={"key": json.dumps(tc.to_json(key))}, json=tc.to_json(value))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.put(endpoint, params={"key": json.dumps(key)}, json=value)
                backoff *= 2

            return response

    async def post(self, path, params={}):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.post(endpoint, json=tc.to_json(params))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.post(endpoint, json=params)
                backoff *= 2

            return response

    async def delete(self, path, key=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            response = await session.delete(endpoint, params={"key": json.dumps(tc.to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(key)})
                backoff *= 2

            return response

    def start(self, **flags):
        return self._host.start(**flags)

    def stop(self):
        return self._host.stop()


class Benchmark(object):
    def __iter__(self):
        for name, item in inspect.getmembers(self):
            if name.startswith('_') or hasattr(Benchmark, name) or not callable(item):
                continue

            yield item

    async def run(self, requests, concurrency):
        responses = []

        start = time.time()
        for i in range(0, len(requests), concurrency):
            response_text = []
            for response in await asyncio.gather(*requests[i:(i + concurrency)]):
                if response.status == 200:
                    response_text.append(response.text())
                else:
                    logging.warning(f"error {response.status}: {await response.text()}")

            responses.extend(await asyncio.gather(*response_text))

        elapsed = time.time() - start
        success = (len(responses) / len(requests)) * 100
        return responses, elapsed, success

    def start(self, **flags):
        pass

    def stop(self):
        pass


async def main(benchmarks):
    import sys

    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument('-k', type=str, help="filter benchmarks to run by name")
    parser.add_argument('--cache_size', type=str, default="2G", help="host cache size")
    parser.add_argument('--concurrency', type=int, help="batch size for concurrent requests")
    parser.add_argument(
        '--num_users', type=int, nargs='+', action='append',
        help="number of unique users to simulate (this flag can be repeated)")

    args = parser.parse_args()

    pattern = args.k
    scales = [n for [n] in args.num_users] if args.num_users else None
    concurrency = args.concurrency
    cache_size = args.cache_size

    for benchmark in benchmarks:
        print(f"running {benchmark}")

        started = False
        for test in benchmark:
            if pattern is None or pattern in test.__name__:
                if not started:
                    benchmark.start(cache_size=cache_size)
                    started = True
                    print()

                for num_users in (scales if scales else benchmark.SCALES):
                    await test(num_users, (concurrency if concurrency else benchmark.CONCURRENCY))

                print()

        if started:
            benchmark.stop()
