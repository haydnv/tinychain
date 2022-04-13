import aiohttp
import argparse
import asyncio
import random
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


async def _run(requests, concurrency=1):
    responses = []

    start = time.time()
    for i in range(0, len(requests), concurrency):
        response_text = []
        for response in await asyncio.gather(*requests[i:(i + concurrency)]):
            if response.status == 200:
                response_text.append(response.text())
            else:
                print(f"error {response.status}: {await response.text()}")

        responses.extend(await asyncio.gather(*response_text))

    elapsed = time.time() - start
    success = (len(responses) / len(requests)) * 100
    return responses, elapsed, success


async def btree_insert(host, num_users):
    link = str(host.link(URI.append("btree_insert")))

    async with aiohttp.ClientSession() as session:
        requests = [session.put(link, json=random.randint(0, 100000)) for _ in range(num_users)]
        responses, elapsed, success = await _run(requests)

    qps = int(num_users / elapsed)
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

        responses, elapsed, success = await _run(requests)

    qps = int((num_keys * concurrent) / elapsed)
    print(f"BTree insert {num_keys} keys x {concurrent} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_upsert(host, concurrent):
    link = str(host.link(URI.append("table_upsert")))

    async with aiohttp.ClientSession() as session:
        keys = list(range(concurrent))
        random.shuffle(keys)

        requests = [session.put(link, json=i) for i in keys]
        responses, elapsed, success = await _run(requests)

    qps = int(concurrent / elapsed)
    print(f"Table insert row x {concurrent} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_read(host, num_users):
    link = str(host.link(URI.append("table_read")))

    async with aiohttp.ClientSession() as session:
        keys = list(range(num_users))
        random.shuffle(keys)

        requests = [session.get(link, params={"key": random.randint(0, 1000)}) for i in keys]
        responses, elapsed, success = await _run(requests, 10)

    qps = int(num_users / elapsed)
    print(f"Table read row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def main(pattern):
    host = start_host("benchmark", Benchmark(), cache_size=CACHE_SIZE, log_level="debug")

    print()
    print("running benchmark")
    print()

    benchmarks = [
        btree_insert,
        btree_multi,
        table_upsert,
        table_read,
    ]

    for benchmark in benchmarks:
        if pattern is None or pattern in benchmark.__name__:
            for num_users in SCALES:
                await benchmark(host, num_users)

            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-k', type=str, help='filter benchmarks to run by name')
    args = parser.parse_args()

    asyncio.run(main(args.k))
