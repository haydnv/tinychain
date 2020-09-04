# TINYCHAIN

 - Python client: [http://github.com/haydnv/tinychain.py](http://github.com/haydnv/tinychain.py)
 - Test suite: [http://github.com/haydnv/tctest](http://github.com/haydnv/tctest)

Tinychain is an application runtime with an integrated database and blockchain support written in
Rust. This is a *preview* release of Tinychain intended to solicit feedback on the concept and API.
Many core features, such as file persistence, are not yet implemented.

Tinychain supports BTree indexes and tables (like MySQL), tensors
(like TensorFlow/Theano/Torch/etc.), graphs (like Neo4j), and blockchains (like Ethereum). Tensor
and graph operations support hardware acceleration on CUDA and OpenCL devices (such as Nvidia GPUs).
Tinychain is intended to bridge the gap between the futuristic distributed computing technology of
blockchain dApps and the security- and performance- critical application stack of traditional
enterprise services.

## Getting started
The easiest way to get started is using the latest Docker image here:
[http://hub.docker.com/repository/docker/haydnv/tinychain](http://hub.docker.com/repository/docker/haydnv/tinychain)

```bash
docker pull haydnv/tinychain:demo1

# include the "--gpus all" argument if you have nvidia-docker installed
docker run -it --rm haydnv/tinychain:demo1 bash

cd /tinychain
source ~/.profile
./target/debug/tinychain --http_port=8702 &

curl -G "http://127.0.0.1:8702/sbin/value/string/ustring"\
  --data-urlencode 'key="Hello, World!"'
```

Alternately, to build from source, you'll need the latest version of Rust and
[ArrayFire](https://arrayfire.org/docs/using_on_linux.htm). Currently Tinychain only supports Linux,
although it has not been tested on Unix or Windows (so it might work without too much effort).
You'll need to set the path to ArrayFire:

```bash
export AF_PATH=/path/to/arrayfire
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH
```

Also update the `env` file with the ArrayFire path in `LD_LIBRARY_PATH`.

Then (if building from source):

```
./env cargo build tinychain
ln -s target/debug/tinychain tinychain
```

Then start Tinychain:

```bash
$ ./tinychain --http_port=8888
```

If you get an error like `error while loading shared libraries: libaf.so.3` it means you
didn't set `AF_PATH` and `LD_LIBRARY_PATH` correctly. This is only necessary for the debug build
of Tinychain.

In a separate terminal, try running this "hello world" program:

```bash
curl -G "http://127.0.0.1:8888/sbin/value/string/ustring" --data-urlencode 'key="Hello, World!"'
```

You can see an up-to-date list of available API endpoints by looking through the
[test suite](http://github.com/haydnv/tctest). More documentation is coming soon.

## Getting involved
Developer documentation is coming soon. Until then, feel free to file a bug for any part
of the Tinychain host software that you want to understand better. Pull requests are also welcome!
