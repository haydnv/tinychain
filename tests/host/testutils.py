import logging
import os
import pathlib
import shutil
import subprocess
import time
import tinychain as tc


TC_PATH = os.getenv("TC_PATH", "host/target/debug/tinychain")
DEFAULT_PORT = 8702


class Process(tc.host.Local.Process):
    """A local TinyChain host process."""

    ADDRESS = "127.0.0.1"

    def __init__(self, path, workspace, force_create=False, libs=[], **flags):
        http_port = flags.get("http_port", DEFAULT_PORT)
        print(f"start host process on port {http_port}")

        # set _process first so it's available to __del__ in case of an exception
        self._process = None

        if http_port is None or not int(http_port) or int(http_port) < 0:
            raise ValueError(f"invalid port: {http_port}")

        if libs and "data_dir" not in flags:
            raise ValueError("hosting a cluster requires specifying a data_dir")

        if "data_dir" in flags:
            maybe_create_dir(flags["data_dir"], force_create)

        args = [
            path,
            f"--workspace={workspace}",
            f"--address={self.ADDRESS}",
        ]

        args.extend(f"--cluster={lib}" for lib in libs)
        args.extend(f"--{flag}={value}" for flag, value in flags.items())

        self._args = args

    def start(self, wait_time):
        """Start this host `Process`."""

        if self._process:
            raise RuntimeError("tried to start a host that's already running")

        self._process = subprocess.Popen(self._args)
        time.sleep(wait_time)

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(f"TinyChain process crashed on startup")
        else:
            logging.info(f"new instance running")

    def stop(self, wait_time=None):
        """Shut down this host `Process`."""

        logging.info(f"Shutting down TinyChain host")
        if self._process:
            self._process.terminate()
            self._process.wait()
            logging.info(f"Host shut down")

            if wait_time:
                time.sleep(wait_time)
        else:
            logging.info(f"Host not running")

        self._process = None

    def __del__(self):
        if self._process:
            self.stop()


def start_host(name, app_or_library=[], overwrite=True, host_uri=None, wait_time=1, **flags):
    print(f"running a local host using the binary at {TC_PATH}")

    if not os.path.isfile(TC_PATH):
        raise RuntimeError(f"invalid executable path: {TC_PATH}")

    deps = tc.app.dependencies(app_or_library) if isinstance(app_or_library, tc.app.Library) else app_or_library

    port = DEFAULT_PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()
    elif deps and tc.uri(deps[0]).port():
        port = tc.uri(deps[0]).port()

    config = []
    for dep in deps:
        dep_config = f"config/{name}"
        dep_config += str(tc.uri(dep).path())

        tc.app.write_config(dep, dep_config, overwrite)
        config.append(dep_config)

    data_dir = f"/tmp/tc/tmp/{port}/{name}"
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    if "log_level" not in flags:
        flags["log_level"] = "debug"

    if "http_port" in flags:
        assert flags["http_port"] == port
    else:
        flags["http_port"] = port

    process = Process(
        TC_PATH,
        workspace=f"/tmp/tc/tmp/{port}/{name}",
        data_dir=data_dir,
        libs=config,
        force_create=True,
        **flags)

    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


def maybe_create_dir(path, force):
    path = pathlib.Path(path)
    if path.exists() and path.is_dir():
        return
    elif force:
        os.makedirs(path)
    else:
        raise RuntimeError(f"no directory at {path}")
