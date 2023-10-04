import os.path
import shutil

import argparse
import asyncio
import inspect
import logging
import random
import time

import rjwt
import tinychain as tc

from .process import Local

DEFAULT_CACHE_SIZE = "8G"
DEFAULT_CONCURRENCY = 5
WORKSPACE = "/tmp/tc/tmp"


def start_local_host(ns, host_uri=None, public_key=None, wait_time=1, **flags):
    assert ns.startswith('/'), f"namespace must be a URI path, not {ns}"
    name = str(ns)[1:].replace('/', '_')

    if not os.path.isfile(TC_PATH):
        hint = "use the TC_PATH environment variable to set the path to the TinyChain host binary"
        raise RuntimeError(f"invalid executable path: {TC_PATH} ({hint})")

    port = DEFAULT_PORT
    if flags.get("http_port"):
        port = flags["http_port"]
        del flags["http_port"]
    elif host_uri is not None and host_uri.port():
        port = host_uri.port()

    if public_key:
        flags["public_key"] = public_key.hex()

    data_dir = f"/tmp/tc/data/{port}/{name}"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    if "log_level" not in flags:
        flags["log_level"] = "debug"

    if "workspace" in flags:
        workspace = flags["workspace"] + f"/{port}/{name}"
        del flags["workspace"]
    else:
        workspace = DEFAULT_WORKSPACE + f"/{port}/{name}"

    process = Local(
        TC_PATH,
        workspace=workspace,
        force_create=True,
        data_dir=data_dir,
        http_port=port,
        **flags)

    process.start(wait_time)
    return tc.host.asyncronous.Local(process, f"http://{process.ADDRESS}:{port}")


class Benchmark(object):
    def __iter__(self):
        for name, item in inspect.getmembers(self):
            if name.startswith('_') or hasattr(Benchmark, name) or not callable(item):
                continue

            yield item

    async def run(self, requests, concurrency):
        requests = [pause_and_run(request) for request in requests]
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

    async def start(self, actor, **flags):
        pass

    def stop(self):
        pass


async def main(benchmarks):
    import sys

    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument('-k', type=str, help="filter benchmarks to run by name")
    parser.add_argument('--cache_size', type=str, default=DEFAULT_CACHE_SIZE, help="host cache size")
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help="concurrent batch size")
    parser.add_argument('--workspace', type=str, default=WORKSPACE, help="workspace directory")
    parser.add_argument(
        '--num_users', type=int, nargs='+', action='append',
        help="number of unique users to simulate (repeat this flag for multiple runs)")

    args = parser.parse_args()

    patterns = set(pattern.strip() for pattern in args.k.split(',')) if args.k else None
    scales = [n for [n] in args.num_users] if args.num_users else None
    concurrency = args.concurrency
    cache_size = args.cache_size
    workspace = args.workspace

    # clean the workspace before running any benchmarks
    if os.path.exists(WORKSPACE):
        shutil.rmtree(WORKSPACE)

    actor = rjwt.Actor('/')

    for benchmark in benchmarks:
        print(f"running {benchmark}")

        started = False
        for test in benchmark:
            if patterns is None or any(pattern in test.__name__ for pattern in patterns):
                if not started:
                    await benchmark.start(actor, cache_size=cache_size, workspace=workspace)
                    started = True
                    print()

                for num_users in (scales if scales else benchmark.SCALES):
                    await test(num_users, concurrency)

                print()

        if started:
            benchmark.stop()
            # clean the workspace again after running a benchmark
            shutil.rmtree(WORKSPACE)


async def pause_and_run(task, pause_length = None):
    pause_length = (random.random() / 10) if pause_length is None else pause_length
    await asyncio.sleep(pause_length)
    return await task