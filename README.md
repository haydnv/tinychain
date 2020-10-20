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

## Contents

 * [Getting Started](#getting-started)
    * [Running Tinychain](#running-tinychain)
    * [The Tinychain Environment](#the-tinychain-environment)
 * [Security](#security)
 * [Protocol](#protocol)
 * [Getting Involved](#getting-involved)

## Getting Started

### Running Tinychain

The easiest way to get started is using the latest Docker image here:
[https://hub.docker.com/r/haydnv/tinychain](https://hub.docker.com/r/haydnv/tinychain)

```bash
docker pull haydnv/tinychain:demo2

# include the "--gpus all" argument if you have nvidia-docker installed
docker run -it --rm haydnv/tinychain:demo2 bash

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
loosely based on TensorFlow v1), testing it, and saving it to a Cluster for users to execute. We'll
go through an example application to see what this means in practice.

Note that these examples are all given in JSON, which is useful if you're implementing a custom
client of your own, but in general it's much easier and more intuitive to use the
[Python client](http://github.com/haydnv/tinychain.py). You can find these examples in Python
[here](https://github.com/haydnv/tctest/blob/master/examples/walkthrough.py) in the test suite.

First, start a new Tinychain host:

```bash
path/to/tinychain --http_port=8702
```

First, use your favorite text editor to create a simple "Hello, World!" program:

```json
{"/sbin/value/string/ustring": "Hello, World!"}
```

Save this program to `myapp.json` and run it with cURL:

```bash
curl "http://127.0.0.1:8702/sbin/transact" -d @myapp.json
```

You should see an output like:

```bash
$ curl "http://127.0.0.1:8702/sbin/transact" -d @myapp.json 
"Hello, World!"
$
```

`/sbin/transact` is an endpoint that will simply attempt to execute whatever JSON you send via
HTTP POST. Other POST endpoints, however, require a more specific executable format--try replacing
the contents of `myapp.json` with this:

```json
[
    ["one", {"/sbin/value/number/int/32": 1}],
    ["two", {"/sbin/value/number/int/32": 2}]
]
```

When you run this, you'll only see:

```json
$ curl "http://127.0.0.1:8702/sbin/transact" -d @myapp.json 
{
  "/sbin/value/number/uint/64": [
    2
  ]
}
```

That's because Tinychain interpreted this as an executable function, and only returned the last
value assigned in the function definition. To return multiple values, we can use a tuple:

```json
[
    ["one", {"/sbin/value/number/int/32": 1}],
    ["two", {"/sbin/value/number/int/32": 2}],
    ["onetwo", {"/sbin/value/tuple": [
        {"$one": []},
        {"$two": []}
    ]}]
]
```

Try it! The value `{"$one": []}` is a `Ref`, meaning a reference to another value in the same
transaction context. `Ref`s are useful for calling object methods--for example, try:

```json
[
    ["one", {"/sbin/value/number/int/32": 1}],
    ["two", {"/sbin/value/number/int/32": 2}],
    ["add_result", {"$one/add": [{"$two": []}]}]
]
```

This should produce:

```bash
$ curl "http://127.0.0.1:8702/sbin/transact" -d @myapp.json 
{
  "/sbin/value/number/int/32": [
    3
  ]
}
```

You can define a function and call it like so:

```json
[
    ["guess", {"/sbin/op/def/get": ["n", [
            ["n_is_five", {"$n/eq": [5]}],
            ["message", {"/sbin/op/ref/if": [{"$n_is_five": []}, "right", "wrong"]}]
        ]]
    }],
    ["outcome", {"$guess": [6]}]
]
```

Tinychain supports generic objects, making it easy to import data into Tinychain from other sources.
For example, consider a third-party service that returns data like
`{"lat": 40.689, "lng": -74.044}`. You can easily import this into Tinychain like so:

```json
{"lat": 40.689, "lng": -74.044}
```

You can easily modify an `Object` to define other instance variables and methods--consider this
example with a new method called `radians`:

```json
{
    "lat": 40.689,
    "lng": -74.044,
    "radians": {"/sbin/op/def/get": [
        ["pi", 3.14159],
        ["radians_per_degree", {"$pi/div": [180]}],
        ["coord_radians", {"/sbin/object": {
            "lat": {"$self/lat/mul": [{"$radians_per_degree": []}]},
            "lng": {"$self/lng/mul": [{"$radians_per_degree": []}]}
        }}]
    ]}
}
```

Of course, you can call an method on a generic `Object` just like any other:

```json
[
    ["greeting", {"/sbin/object": {
        "en": "Hello!",
        "es": "¡Hola!",
        "render": {"/sbin/op/def/get": ["lang", [
            ["is_spanish", {"$lang/eq": ["es"]}],
            ["rendered", {"/sbin/op/ref/if": [{"$is_spanish": []}, {"$self/es": []}, {"$self/en": []}]}]
        ]]}
    }}],
    ["result", {"$greeting/render": ["es"]}]
]
```

This is handy for prototyping and debugging, but for production use you'll want to impose more
formal constraints on your data. You can do this by using a `Class`. Classes in Tinychain work
just like any object-oriented language; you can think of them as formal constraints on data which
enable the application handling the data to make very specific assumptions. For example, this
`Class` called `Degree` extends `Number` and implements a method called `radians`.

```json
[
    ["Degree", {"/sbin/value/number": {
        "radians": {"/sbin/op/def/get": [
            ["pi", 3.14159],
            ["radians_per_degree", {"$pi/div": [180]}],
            ["radians", {"$self/mul": [{"$radians_per_degree": []}]}]
        ]}
    }}],
    ["d", {"$Degree": [90]}],
    ["r", {"$d/radians": []}]
]
```

** This feature is not yet implemented

## Security

Tinychain will use [OAuth2](https://oauth.net/2/) with [JSON Web Tokens](https://oauth.net/2/jwt/)
to create a mutually intelligible system of authorization, ownership, and encryption. This is
planned for implementation in 2020Q4.

## Protocol

All Tinychain primitives are expressible in JSON. The basic formats are:

```javascript
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

## Getting Involved
Developer documentation is coming soon. Until then, feel free to file a bug for any part
of the Tinychain host software that you want to understand better. Pull requests are also welcome!
