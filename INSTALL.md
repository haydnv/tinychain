# Client

Most users will only need to run the TinyChain Python client, not a TinyChain host. You can install the client with `pip` by running `pip3 install tinychain` (on very new systems where Python 3 is the default this may be `pip install tinychain`).

Verify that you have the Python client installed correctly by running this test script:

```python
import tinychain as tc

host = tc.host.Host("http://demo.tinychain.net")
assert host.get(tc.uri(tc.String), "Hello, World!") == "Hello, World!"
print("success!")
```

# Host

## Easy install

The quick and easy way to get TinyChain up and running to try it out is to use Docker:

```bash
# build the Dockerfile from the GitHub repo, then run a new container with TinyChain listening on host port 8702
# the "-it" option also opens an interactive terminal
docker run -it -p 8702:8702/tcp $(docker build https://github.com/haydnv/tinychain.git -q) ./tinychain --address=0.0.0.0
```

You can check that your installation succeeded by loading `http://127.0.0.1:8702/state/scalar/value/string?key="Hello, World!"` in your browser.

## Automatic install (Ubuntu)

An install script is provided for Ubuntu (only tested on Ubuntu 20.04):

```bash
curl https://raw.githubusercontent.com/haydnv/tinychain/master/install.sh -sSf | bash
```

## Manual install (Ubuntu)

1. If you need CUDA support for GPU acceleration, first install CUDA 11 by following the instructions here:
[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation).
If you're not sure, skip this step.
2. Install ArrayFire by following the instructions here: [https://github.com/arrayfire/arrayfire/wiki/Install-ArrayFire-From-Linux-Package-Managers](https://github.com/arrayfire/arrayfire/wiki/Install-ArrayFire-From-Linux-Package-Managers)
3. Install cargo by following the instructions here: [https://doc.rust-lang.org/cargo/getting-started/installation.html](https://doc.rust-lang.org/cargo/getting-started/installation.html)
4. Install TinyChain by running `cargo install tinychain --features=tensor`

## Manual install (other OS)

Installation on other operating systems has not been tested and may not be straightforward. If you need to install TinyChain on an operating system other than 64-bit x86 Ubuntu Linux, please [open an issue](https://github.com/haydnv/tinychain/issues).

1. If you need CUDA support for GPU acceleration, make sure to install CUDA first. If you're not sure, skip this step.
2. Install ArrayFire by following the instructions here: [https://arrayfire.org/docs/installing.htm](https://arrayfire.org/docs/installing.htm)
3. Install cargo by following the instructions here: [https://doc.rust-lang.org/cargo/getting-started/installation.html](https://doc.rust-lang.org/cargo/getting-started/installation.html)
4. Install TinyChain by running `cargo install tinychain --features=tensor`


Tip: the ArrayFire library requires the environment variables `AF_PATH` and `LD_LIBRARY_PATH` to be set at build time **and** at run time.
The vast majority of installation failures happen as a result of missing or incorrect environment variables.
If you have any problems with your TinyChain installation, the first thing to check is that `AF_PATH` is set to the ArrayFire installation root directory (usually `/opt/arrayfire`)
and included in `LD_LIBRARY_PATH` (e.g. by running `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib64`).

