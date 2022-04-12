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

    @tc.post
    def insert_btree(self, num_keys: tc.UInt):
        tc.Stream.range(num_keys).for_each(tc.get(lambda i: self.btree.insert([i])))


if __name__ == "__main__":
    host = start_host("benchmark", Benchmark(), cache_size=CACHE_SIZE)

    print()
    print("running benchmark")
    print()

    print("benchmarking BTree")

    num_keys = 1_000_000
    start = time.time()
    host.post(URI.append("insert_btree"), {"num_keys": num_keys})
    elapsed = time.time() - start
    print(f"BTree insert {num_keys} keys: {elapsed}")
