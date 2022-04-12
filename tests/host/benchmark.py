import random

import aiohttp
import asyncio
import time
import tinychain as tc

from testutils import start_host

CACHE_SIZE = "8G"
URI = tc.URI("/app/benchmark")


class Benchmark(tc.app.App):
    __uri__ = URI

    def __init__(self):
        self.btree = tc.chain.Sync(tc.btree.BTree([("i", tc.I32)]))
        tc.app.App.__init__(self)

    @tc.put
    def btree_insert(self, value: tc.UInt):
        return self.btree.insert([value])

    @tc.post
    def btree_multi(self, start: tc.UInt, stop: tc.UInt):
        return tc.Stream.range((start, stop)).for_each(tc.get(lambda i: self.btree.insert([i])))


async def btree_insert(host, concurrent):
    link = str(host.link(URI.append("btree_multi")))

    async with aiohttp.ClientSession() as session:
        ops = [session.put(link, json=random.randint(0, 100000)) for _ in range(concurrent)]

        start = time.time()
        for response in await asyncio.gather(*ops):
            await response.text()

        elapsed = time.time() - start

    qps = int(concurrent / elapsed)
    print(f"BTree insert single key x {concurrent} users: {elapsed:.4f} seconds ({qps} QPS)")


async def btree_multi(host, concurrent):
    i = 0
    num_keys = 1000
    link = str(host.link(URI.append("btree_multi")))
    async with aiohttp.ClientSession() as session:
        ops = []
        for _ in range(concurrent):
            ops.append(session.post(link, json={"start": i, "stop": i + num_keys}))
            i += num_keys

        start = time.time()
        for response in await asyncio.gather(*ops):
            await response.text()
        elapsed = time.time() - start

    qps = int((num_keys * concurrent) / elapsed)
    print(f"BTree insert {num_keys} keys x {concurrent} users: {elapsed:.4f} seconds ({qps} QPS)")


async def main():
    host = start_host("benchmark", Benchmark(), cache_size=CACHE_SIZE, log_level="debug")

    print()
    print("running benchmark")
    print()

    print("benchmarking BTree")
    print()

    for concurrency in [1, 5, 10, 100, 1000, 10000]:
        await btree_insert(host, concurrency)

    print()

    for concurrency in [1, 5, 10, 100, 1000]:
        await btree_multi(host, concurrency)


if __name__ == "__main__":
    asyncio.run(main())
