import aiohttp
import argparse
import asyncio
import json
import random
import testutils
import time
import tinychain as tc

URI = tc.URI("/app/benchmark")
SCALES = [1, 5, 10, 100]
DEFAULT_CONCURRENCY = 5
BACKOFF = 0.01


# TODO: move this class into a new async module in tc.host
class Host(object):
    RETRYABLE = [409, 502, 503]

    def __init__(self, host):
        self.__uri__ = tc.uri(host)
        self._host = host  # keep a reference here so it doesn't get dropped

    async def get(self, path, key=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession() as session:
            response = await session.get(endpoint, params={"key": json.dumps(tc.to_json(key))})

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.get(endpoint, params={"key": json.dumps(key)})
                backoff *= 2

            return response

    async def put(self, path, key=None, value=None):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession() as session:
            response = await session.put(endpoint, params={"key": json.dumps(tc.to_json(key))}, json=tc.to_json(value))

            backoff = BACKOFF
            while response.status in self.RETRYABLE:
                await asyncio.sleep(backoff)
                response = await session.put(endpoint, params={"key": json.dumps(key)}, json=value)
                backoff *= 2

            return response

    async def post(self, path, params={}):
        endpoint = str(tc.uri(self) + path)

        async with aiohttp.ClientSession() as session:
            response = await session.post(endpoint, json=tc.to_json(params))

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

        self.tensor1 = tc.chain.Sync(tc.tensor.Dense(([100, 50], tc.F32)))
        self.tensor2 = tc.chain.Sync(tc.tensor.Dense(([50, 1], tc.F32)))

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

    @tc.get
    def tensor_multiply(self) -> tc.F32:
        return (self.tensor1 * self.tensor2.transpose()).sum()

    @tc.get
    def train_neural_net(self, cxt) -> tc.F32:
        cxt.inputs = tc.tensor.Dense.random_uniform([20, 2], 0, 1)
        cxt.labels = cxt.inputs[:, 0].logical_xor(cxt.inputs[:, 1]).expand_dims().copy()

        def cost(_i, o):
            return (o - cxt.labels)**2

        layers = [
            tc.ml.nn.DNNLayer.create(2, 2, tc.ml.sigmoid),
            tc.ml.nn.DNNLayer.create(2, 1, tc.ml.sigmoid)]

        dnn = tc.ml.nn.Sequential(layers)
        cxt.optimizer = tc.ml.optimizer.Adam(dnn, cost)

        @tc.post
        def cond(loss: tc.tensor.Dense):
            return (abs(loss) > 0.5).any()

        @tc.closure(cxt.optimizer, cxt.inputs)
        @tc.post
        def train(step: tc.U32):
            loss = cxt.optimizer.train(step, cxt.inputs)
            return {"step": step + 1, "loss": loss}

        cxt.training = tc.While(cond, train, {"step": 1, "loss": cxt.labels - dnn.eval(cxt.inputs)})
        return tc.After(cxt.training, (cxt.optimizer.ml_model.eval(cxt.inputs) == cxt.labels).all())


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
    requests = [host.put(URI + "/btree_insert", value=random.randint(0, 100000)) for _ in range(num_users)]
    responses, elapsed, success = await _run(requests, concurrency)
    qps = int(num_users / elapsed)
    print(f"BTree insert key x {num_users} users: {elapsed:.4f} seconds @ {qps}/s, {success:.2f}% success")


async def btree_multi(host, num_users, concurrency):
    i = 0
    num_keys = 1000

    requests = []
    for _ in range(num_users):
        requests.append(host.post(URI + "/btree_multi", {"start": i, "stop": i + num_keys}))
        i += num_keys

    responses, elapsed, success = await _run(requests, concurrency)

    qps = int((num_keys * num_users) / elapsed)
    print(f"BTree insert {num_keys} keys x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_upsert(host, num_users, concurrency):
    keys = list(range(num_users))
    random.shuffle(keys)

    requests = [host.put(URI + "/table_upsert", value=i) for i in keys]
    responses, elapsed, success = await _run(requests, concurrency)

    qps = int(num_users / elapsed)
    print(f"Table insert row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def table_read(host, num_users, concurrency):
    keys = list(range(num_users))
    random.shuffle(keys)

    requests = [host.get(URI + "/table_read", i) for i in keys]
    responses, elapsed, success = await _run(requests, concurrency)

    qps = int(num_users / elapsed)
    print(f"Table read row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def tensor_multiply(host, num_users, concurrency):
    requests = [host.get(URI + "/tensor_multiply") for _ in range(num_users)]
    responses, elapsed, success = await _run(requests, concurrency)
    qps = int(num_users / elapsed)
    print(f"Tensor transpose, broadcast, multiply & sum x {num_users} users: "
          + f"{elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def train_neural_net(host, num_users, concurrency):
    requests = [host.get(URI + "/tensor_multiply") for _ in range(num_users)]
    responses, elapsed, success = await _run(requests, concurrency)
    qps = int(num_users / elapsed)
    print(f"create & train a neural net x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


async def main(pattern, scales, concurrency, cache_size):
    host = testutils.start_host("benchmark", Benchmark(), cache_size=cache_size)
    host = Host(host)

    print()
    print("running benchmark")
    print()

    benchmarks = [
        btree_insert,
        btree_multi,
        table_upsert,
        table_read,
        tensor_multiply,
        train_neural_net,
    ]

    for benchmark in benchmarks:
        if pattern is None or pattern in benchmark.__name__:
            for num_users in scales:
                await benchmark(host, num_users, concurrency)

            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument('-k', type=str, help="filter benchmarks to run by name")
    parser.add_argument('--cache_size', type=str, default="2G", help="host cache size")
    parser.add_argument('--concurrency', type=int, default=10, help="batch size for concurrent requests")
    parser.add_argument(
        '--num_users', type=int, nargs='+', action='append',
        help="number of unique users to simulate (this flag can be repeated)")

    args = parser.parse_args()

    num_users = [n for [n] in args.num_users] if args.num_users else SCALES
    asyncio.run(main(args.k, num_users, args.concurrency, args.cache_size))
