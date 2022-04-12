import random

import aiohttp
import asyncio
import time
import tinychain as tc

from testutils import start_host

CACHE_SIZE = "8G"
URI = tc.URI("/app/benchmark")
SCALES = [1, 5, 10, 100]


class Benchmark(tc.app.App):
    __uri__ = URI

    def __init__(self):
        self.btree = tc.chain.Sync(tc.btree.BTree([("i", tc.I32)]))

        table_schema = tc.table.Schema([("x", tc.I32)], [("y", tc.I32)]).create_index("y", ["y"])
        self.table = tc.chain.Sync(tc.table.Table(table_schema))

        tc.app.App.__init__(self)

    @tc.get
    def btree_read(self, key: tc.UInt):
        return self.btree[key]

    @tc.put
    def btree_insert(self, value: tc.UInt):
        return self.btree.insert([value])

    @tc.post
    def btree_multi(self, start: tc.UInt, stop: tc.UInt):
        return tc.Stream.range((start, stop)).for_each(tc.get(lambda i: self.btree.insert([i])))

    @tc.put
    def table_upsert(self, value: tc.UInt):
        return self.table.upsert([value], [value])

    @tc.get
    def table_read(self, key: tc.UInt):
        return self.table[(key,)]


async def _run(requests):
    responses = []

    start = time.time()
    for request in requests:
        # TODO: batch requests
        response = await request
        if response.status == 200:
            responses.append(await response.text())
        else:
            print(f"error {response.status}: {await response.text()}")

    elapsed = time.time() - start
    return responses, elapsed


async def btree_insert(host, num_users):
    link = str(host.link(URI.append("btree_insert")))

    async with aiohttp.ClientSession() as session:
        requests = [session.put(link, json=random.randint(0, 100000)) for _ in range(num_users)]
        responses, elapsed = await _run(requests)

    qps = int(num_users / elapsed)
    success = (len(responses) / len(requests)) * 100
    print(f"BTree insert key x {num_users} users: {elapsed:.4f} seconds @ {qps}/s, {success:.2f}% success")


async def btree_multi(host, concurrent):
    i = 0
    num_keys = 1000
    link = str(host.link(URI.append("btree_multi")))
    async with aiohttp.ClientSession() as session:
        requests = []
        for _ in range(concurrent):
            requests.append(session.post(link, json={"start": i, "stop": i + num_keys}))
            i += num_keys

        responses, elapsed = await _run(requests)

    success = (len(responses) / len(requests)) * 100
    qps = int((num_keys * concurrent) / elapsed)
    print(f"BTree insert {num_keys} keys x {concurrent} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def btree_read(host, num_users):
    link = str(host.link(URI.append("btree_read")))

    async with aiohttp.ClientSession() as session:
        requests = [session.get(link, json=random.randint(0, 1000)) for _ in range(num_users)]
        responses, elapsed = await _run(requests)

    qps = int(num_users / elapsed)
    success = (len(responses) / len(requests)) * 100
    print(f"BTree read key x {num_users} users: {elapsed:.4f} seconds @ {qps}/s, {success:.2f}% success")


async def table_upsert(host, concurrent):
    link = str(host.link(URI.append("table_upsert")))

    async with aiohttp.ClientSession() as session:
        keys = list(range(concurrent))
        random.shuffle(keys)

        requests = [session.put(link, json=i) for i in keys]
        responses, elapsed = await _run(requests)

    success = (len(responses) / len(requests)) * 100
    qps = int(concurrent / elapsed)
    print(f"Table insert row x {concurrent} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_read(host, concurrent):
    link = str(host.link(URI.append("table_read")))

    async with aiohttp.ClientSession() as session:
        keys = list(range(concurrent))
        random.shuffle(keys)

        requests = [session.get(link, params={"key": random.randint(0, 1000)}) for i in keys]
        responses, elapsed = await _run(requests)

    success = (len(responses) / len(requests)) * 100
    qps = int(concurrent / elapsed)
    print(f"Table read row x {concurrent} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def main():
    host = start_host("benchmark", Benchmark(), cache_size=CACHE_SIZE, log_level="debug")

    print()
    print("running benchmark")
    print()

    print("benchmarking BTree")
    print()

    for concurrency in SCALES:
        await btree_insert(host, concurrency)

    print()

    for concurrency in SCALES:
        await btree_multi(host, concurrency)

    print()

    for concurrency in SCALES:
        await btree_read(host, concurrency)

    print("benchmarking Table")
    print()

    for concurrency in SCALES:
        await table_upsert(host, concurrency)

    print()

    for concurrency in SCALES:
        await table_read(host, concurrency)


if __name__ == "__main__":
    asyncio.run(main())
