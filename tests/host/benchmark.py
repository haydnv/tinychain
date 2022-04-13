import aiohttp
import argparse
import asyncio
import json
import random
import time
import tinychain as tc

URI = tc.URI("/app/benchmark")
SCALES = [1, 5, 10, 100]
DEFAULT_CONCURRENCY = 5
BACKOFF = 0.01


# TODO: merge this with tc.host.Host to allow async requests
class Host(object):
    RETRYABLE = [409, 502, 503]

    def __init__(self, uri):
        self.__uri__ = uri

    async def get(self, path, key=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession() as session:
            response = await session.get(endpoint, params={"key": json.dumps(key)})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(key)})
                backoff *= 2

            return response

    async def put(self, path, key=None, value=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession() as session:
            response = await session.put(endpoint, params={"key": json.dumps(key)}, json=value)

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.put(endpoint, params={"key": json.dumps(key)}, json=value)
                backoff *= 2

            return response

    async def post(self, path, params={}):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession() as session:
            response = await session.post(endpoint, json=params)

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.post(endpoint, json=params)
                backoff *= 2

            return response


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


async def _run(requests, concurrency):
    responses = []

    start = time.time()
    for i in range(0, len(requests), concurrency):
        response_text = []
        for response in await asyncio.gather(*requests[i:(i + concurrency)]):
            if response.status == 200:
                response_text.append(response.text())
            else:
                print(f"error {response.status}: {await response.text()}")
                exit()

        responses.extend(await asyncio.gather(*response_text))

    elapsed = time.time() - start
    success = (len(responses) / len(requests)) * 100
    return responses, elapsed, success


async def btree_insert(host, num_users, concurrency):
    requests = [host.put("/btree_insert", value=random.randint(0, 100000)) for _ in range(num_users)]
    responses, elapsed, success = await _run(requests, concurrency)
    qps = int(num_users / elapsed)
    print(f"BTree insert key x {num_users} users: {elapsed:.4f} seconds @ {qps}/s, {success:.2f}% success")


async def btree_multi(host, num_users, concurrency):
    i = 0
    num_keys = 1000

    requests = []
    for _ in range(num_users):
        requests.append(host.post("/btree_multi", {"start": i, "stop": i + num_keys}))
        i += num_keys

    responses, elapsed, success = await _run(requests, concurrency)

    qps = int((num_keys * num_users) / elapsed)
    print(f"BTree insert {num_keys} keys x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_upsert(host, num_users, concurrency):
    keys = list(range(num_users))
    random.shuffle(keys)

    requests = [host.put("/table_upsert", value=i) for i in keys]
    responses, elapsed, success = await _run(requests, concurrency)

    qps = int(num_users / elapsed)
    print(f"Table insert row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_read(host, num_users, concurrency):
    keys = list(range(num_users))
    random.shuffle(keys)

    requests = [host.get("/table_read", i) for i in keys]
    responses, elapsed, success = await _run(requests, concurrency)

    qps = int(num_users / elapsed)
    print(f"Table read row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def main(pattern, concurrency):
    app = Benchmark()
    app_path = "config/benchmark" + str(tc.uri(app).path())
    tc.app.write_config(app, app_path, overwrite=True)
    print(f"wrote config for {app} to {app_path}")

    host = Host(tc.URI("http://127.0.0.1:8702") + tc.uri(Benchmark))

    try:
        assert await host.get("/")
    except Exception as e:
        print(f"could not contact host at {tc.uri(host)}: {e}")
        print("is there a TinyChain host running on this port? if not, you can run a command like this to start one:")
        print(" ".join([
            "host/target/release/tinychain",
            "--address=127.0.0.1",
            f"--cluster={app_path}",
            "--workspace=/tmp/tc/tmp/8702/benchmark/tmp",
            "--data_dir=/tmp/tc/tmp/benchmark",
            "--http_port=8702",
        ]))
        exit()

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
                await benchmark(host, num_users, concurrency)

            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-k', type=str, help='filter benchmarks to run by name')
    parser.add_argument('--concurrency', type=int, default=10, help='batch size for concurrent requests')
    args = parser.parse_args()

    asyncio.run(main(args.k, args.concurrency))
