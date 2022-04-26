import sys

import aiohttp
import argparse
import asyncio
import inspect
import json
import logging
import random
import time
import tinychain as tc

from .. import benchmark
from ..process import start_local_host


class DataStructures(tc.app.App):
    __uri__ = tc.URI("http://127.0.0.1:8702") + "/app/benchmark/data_structures"

    ml = tc.ml.service.ML

    def __init__(self, chain_type):
        self.btree = chain_type(tc.btree.BTree([("i", tc.I32)]))

        table_schema = tc.table.Schema([("x", tc.I32)], [("y", tc.I32)]).create_index("y", ["y"])
        self.table = chain_type(tc.table.Table(table_schema))

        self.tensor1 = chain_type(tc.tensor.Dense(([100, 50], tc.F32)))
        self.tensor2 = chain_type(tc.tensor.Dense(([50, 1], tc.F32)))

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
    def neural_net_train(self, cxt) -> tc.F32:
        cxt.inputs = tc.tensor.Dense.random_uniform([20, 2], 0, 1)
        cxt.labels = cxt.inputs[:, 0].logical_xor(cxt.inputs[:, 1]).expand_dims().copy()

        def cost(i, o):
            labels = i[:, 0].logical_xor(i[:, 1]).expand_dims()
            return (o - labels)**2

        layers = [
            tc.ml.nn.Linear.create(2, 2, tc.ml.sigmoid),
            tc.ml.nn.Linear.create(2, 1, tc.ml.sigmoid)]

        dnn = tc.ml.nn.Sequential(layers)
        cxt.optimizer = tc.ml.optimizer.Adam(dnn, cost)

        return tc.After(
            cxt.optimizer.train(1, cxt.inputs),
            (abs(cxt.labels - cxt.optimizer.ml_model.eval(cxt.inputs)) >= 0.5).cast(tc.U8).sum())


class ConcurrentWriteBenchmarks(benchmark.Benchmark):
    CONCURRENCY = 10
    SCALES = [1, 5, 10, 100]

    def __init__(self):
        self._base_path = tc.uri(DataStructures).path()
        self.host = None

    def __repr__(self):
        return "concurrent write benchmarks"

    def start(self, **flags):
        host = start_local_host("benchmark_writes", [tc.ml.service.ML(), DataStructures(tc.chain.Sync)], **flags)
        self.host = benchmark.Host(host)

    def stop(self):
        if self.host:
            self.host.stop()

    def _link(self, path):
        return self._base_path + path

    async def btree_insert(self, num_users, concurrency):
        requests = [self.host.put(self._link("/btree_insert"), value=random.randint(0, 100000)) for _ in range(num_users)]
        responses, elapsed, success = await self.run(requests, concurrency)
        qps = int(num_users / elapsed)
        print(f"BTree insert key x {num_users} users: {elapsed:.4f} seconds @ {qps}/s, {success:.2f}% success")
        await self.host.delete(self._link("/btree"))

    async def btree_multi(self, num_users, concurrency):
        i = 0
        num_keys = 1000

        requests = []
        for _ in range(num_users):
            requests.append(self.host.post(self._link("/btree_multi"), {"start": i, "stop": i + num_keys}))
            i += num_keys

        responses, elapsed, success = await self.run(requests, concurrency)

        qps = int((num_keys * num_users) / elapsed)
        print(f"BTree insert {num_keys} keys x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")
        await self.host.delete(self._link("/btree"))

    async def table_upsert(self, num_users, concurrency):
        keys = list(range(num_users))
        random.shuffle(keys)

        requests = [self.host.put(self._link("/table_upsert"), value=i) for i in keys]
        responses, elapsed, success = await self.run(requests, concurrency)

        qps = int(num_users / elapsed)
        print(f"Table insert row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")
        await self.host.delete(self._link("/table"))

    async def neural_net_train(self, num_users, concurrency):
        requests = [self.host.get(self._link("/neural_net_train")) for _ in range(num_users)]
        responses, elapsed, success = await self.run(requests, concurrency)
        qps = num_users / elapsed
        print(f"create & train a neural net x {num_users} users: {elapsed:.4f}s @ {qps:.2f}/s, {success:.2f}% success")


class LoadBenchmarks(benchmark.Benchmark):
    CONCURRENCY = 10
    SCALES = [1, 5, 10, 100, 1000]

    def __init__(self):
        self._base_path = tc.uri(DataStructures).path()
        self.host = None

    def __repr__(self):
        return "load benchmarks"

    def start(self, **flags):
        host = start_local_host("benchmark_load", DataStructures(tc.chain.Sync), **flags)
        self.host = benchmark.Host(host)

    def stop(self):
        if self.host:
            return self.host.stop()

    def _link(self, path):
        return self._base_path + path

    async def table_read(self, num_users, concurrency):
        keys = list(range(num_users))
        random.shuffle(keys)

        requests = [self.host.get(self._link("/table_read"), i) for i in keys]
        responses, elapsed, success = await self.run(requests, concurrency)

        qps = int(num_users / elapsed)
        print(f"Table read row x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")

    async def tensor_multiply(self, num_users, concurrency):
        requests = [self.host.get(self._link("/tensor_multiply")) for _ in range(num_users)]
        responses, elapsed, success = await self.run(requests, concurrency)
        qps = int(num_users / elapsed)
        print(f"Tensor transpose, broadcast, multiply & sum x {num_users} users: "
              + f"{elapsed:.4f}s @ {qps}/s, {success:.2f}% success")


class ReplicationBenchmarks(benchmark.Benchmark):
    CONCURRENCY = 10
    SCALES = [1, 5, 10, 100]
    NUM_HOSTS = 4

    def __init__(self):
        self._base_path = tc.uri(DataStructures).path()
        self.hosts = []

    def __repr__(self):
        return f"replication benchmarks with {self.NUM_HOSTS} hosts"

    def start(self, **flags):
        default_port = tc.uri(DataStructures).port()
        assert default_port

        for i in range(self.NUM_HOSTS):
            port = default_port + i
            host_uri = tc.URI(f"http://127.0.0.1:{port}/")
            host = start_local_host(
                "benchmark_load", DataStructures(tc.chain.Block), overwrite=True, host_uri=host_uri, **flags)

            self.hosts.append(benchmark.Host(host))

    def stop(self):
        for host in self.hosts:
            return host.stop()

    def _link(self, path):
        return self._base_path + path

    async def blockchain_table_write(self, num_users, concurrency):
        requests = [
            self.hosts[0].put(self._link("/table_upsert"), value=random.randint(0, 10000))
            for _ in range(num_users)]

        responses, elapsed, success = await self.run(requests, concurrency)

        qps = int(num_users / elapsed)
        print(f"blockchain Table upsert x {num_users} users: {elapsed:.4f}s @ {qps}/s, {success:.2f}% success")

        print("stop replica...")
        self.hosts[1].stop()

        print("modifying chain contents--you'll see a 'bad gateway' error as the cluster discovers the stopped replica")
        response = await self.hosts[0].put(self._link("/table_upsert"), value=random.randint(0, 10000))
        assert response.status == 200

        start = time.time()
        wait_time = 0.5
        self.hosts[1].start(wait_time=wait_time)
        elapsed = time.time() - start

        print(f"replica rejoin time w/ full table reconstruction, including {wait_time}s startup time: {elapsed:.2f}s")

        await self.hosts[0].delete(self._link("/table"))


if __name__ == "__main__":
    task = benchmark.main([
        ConcurrentWriteBenchmarks(),
        LoadBenchmarks(),
        ReplicationBenchmarks(),
    ])

    asyncio.run(task)
