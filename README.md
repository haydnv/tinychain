# TINYCHAIN

**Sign up for updates at [https://rb.gy/3h1bws](https://rb.gy/3h1bws)!**

 - Docker image: [https://hub.docker.com/r/haydnv/tinychain](https://hub.docker.com/r/haydnv/tinychain)
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
[https://hub.docker.com/r/haydnv/tinychain](https://hub.docker.com/r/haydnv/tinychain)

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
$ ./tinychain --http_port=8702
```

If you get an error like `error while loading shared libraries: libaf.so.3` it means you
didn't set `AF_PATH` and `LD_LIBRARY_PATH` correctly. This is only necessary for the debug build
of Tinychain.

In a separate terminal, try running this "hello world" program:

```bash
curl -G "http://127.0.0.1:8702/sbin/value/string/ustring" --data-urlencode 'key="Hello, World!"'
```

You can see an up-to-date list of available API endpoints by looking through the
[test suite](http://github.com/haydnv/tctest). More documentation is coming soon.

### The Tinychain Environment

Tinychain is both a database platform (like MySQL) and an analytics/ML platform (like TensorFlow)
as well as a Turing-complete runtime (like Ethereum) with blockchain-powered versioning features.**
In general, Tinychain development consists of constructing a deferred execution graph (very
loosely based on TensorFlow v1), testing it, and saving it to a [Cluster](#cluster) for users to
execute. We'll go through an example application to see what this means in practice.

First, start a new Tinychain host:

```bash
path/to/tinychain --http_port=8702
```

First, use your favorite text editor to create a simple "Hello, World!" program:

```json
{"/sbin/value/string/ustring": "Hello, World!"}
```

We'll go over the particulars of this exact format in a moment. For now, save this program to
`myapp.json` and use cURL to run the program:

```bash
curl "http://127.0.0.1:8702/sbin/transact" -d @myapp.json
```


## Security

Tinychain will use [OAuth2](https://oauth.net/2/) with [JSON Web Tokens](https://oauth.net/2/jwt/)
to create a mutually intelligible system of authorization, ownership, and encryption. This is
planned for implementation in 2020Q4.

## Protocol

All Tinychain primitives are expressible in JSON. The basic formats are:

```
// a Link to "/path/to/subject", to be resolved by peer discovery
{"/path/to/subject": []}

// a Link to "/path/to/subject" which must be resolved specifically by http://tinychain.net 
{"http://tinychain.net/path/to/subject": []}

// a ValueId, part of an assignment
{"value_id": []}

// a Ref (reference to an assigned value in the same Transaction context)
{"$value_id": []}

// a reference to a GET Op has exactly one argument
{"/path/to/subject/op_name": ["key"]}

// a reference to a GET Method has exactly one argument
{"$subject/method_name": ["key"]}

// a reference to a PUT Op has exactly two arguments
{"/path/to/subject/op_name": ["key", "value"]}

// a reference to a PUT Method has exactly two arguments
{"$subject/method_name": ["key", "value"]}
```

There is an exception for GET Ops starting with `/sbin/value`: these are resolved immediately when
parsed--for example `{"/sbin/value/number/uint/8": [0]}` is a `uint/8`, not a GET Op. For
convenience, Tinychain will interpret JSON strings as ValueIds in contexts when a ValueId is
specifically expected (e.g., to define a POST Op).

Tinychain supports three HTTP access methods:

 * **GET**: supports reading, but not writing, mutable state. A GET Op can be executed via HTTP
without specifying a transaction context; if the subject of the resource is part of a Cluster, the
Cluster will return the state of the subject as of the latest commit. Via HTTP, GET takes one
query parameter, called "key,", as in http://127.0.0.1:8888/sbin/value/number/int/32?key=1.
 * **PUT**: supports writing, but not reading, mutable state. A PUT Op can be executed via HTTP
without specifying a transaction context; if the subject of the resource is part of a Cluster, the
Cluster will take the state to be its state as of the latest commit. Via HTTP, PUT takes one
query parameter, called "key,", as in http://127.0.0.1:8888/app/vendor/inventory?key=[1]; it also
requires a value argument, which is the body of the request.
 * **POST**: supports reading and writing mutable state and must return exactly one value.

## Getting involved
Developer documentation is coming soon. Until then, feel free to file a bug for any part
of the Tinychain host software that you want to understand better. Pull requests are also welcome!
