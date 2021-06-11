
Python API documentation: [http://tinychain.readthedocs.io/en/latest/](http://tinychain.readthedocs.io/en/latest/)

## Getting started

This guide assumes that you are familiar with the basic concepts of hosting a Tinychain service. If not, please read the Tinychain [project README](http://github.com/haydnv/tinychain) first.

You can install the Python client using Pip:

```bash
pip3 install tinychain
```

If you don't have a Tinychain host already available for testing, you'll also need to download the [latest release](http://github.com/haydnv/tinychain/releases) from GitHub. Note that binary releases are only available for 64-bit x86 Linux at this time. If you have `cargo` installed, you can compile and install Tinychain for your platform by running `cargo install tinychain`. Using `cargo install` should add the `tinychain` command to your `PATH`, so you can just set `TC_PATH = "tinychain"` in the "Hello, World!" example below.

To check that everything is working, try this "Hello, World!" program:

```python
import tinychain as tc

TC_PATH = "/path/to/tinychain"  # <-- edit this!
WORKSPACE = "/tmp/tc/workspace" # <-- optionally, edit this also
ENDPOINT = "/transact/hypothetical"

if __name__ == "__main__":
   host = tc.host.Local(TC_PATH, WORKSPACE)
   print(host.post(ENDPOINT, tc.String("Hello, World!")))
```

## Overview

The Tinychain client provides a developer-friendly API to build Tinychain transactions (i.e., distributed compute graphs) using the familiar Python language, but without any of the associated restrictions on performance or locality. A Tinychain transaction can span any number of networked hosts and automatically takes advantage of the hosts' concurrent and parallel computing resources (like multiple cores and GPU acceleration) without any extra application code. A Tinychain service defined using the Python client can be packaged using familiar distribution mechanisms like [Pip](http://pypi.org/project/pip) so that other Tinychain developers can use it just like a regular Python developer would use a regular Python library. This can save your clients a lot of time and hassle when integrating your cloud API into their service.

Every value in a Tinychain service is either a `State`, representing a Tinychain state like a `Number` or a `Chain`, or a `Ref`, which tells a Tinychain transaction how to access or calculate a particular `State`. For example:

```python
@tc.get_op
def example(txn) -> tc.Number:
    txn.a = tc.Number(5) # this is a State
    txn.b = tc.Number(10) # this is a State
    txn.product = txn.a * txn.b # this is a Ref
    return txn.product
```

Notice the assignments to the transaction context `txn`. This is necessary to assign an addressable name to a `State`, in order to read its value or call an instance method. For example, calling `tc.Number(5) * tc.Number(10)` above would result in an error, because `Number.mul` is an instance method, and `tc.Number(5)` is not addressable. When the state `tc.Number(5)` is assigned to `txn.a`, it becomes addressable within the transaction context as `$a`, making it possible to resolve `OpRef.Get("$a/mul", "$b")`.

The constructor of a `State` always takes exactly one argument, which is the form of that `State`. For example, `tc.Number(3)` constructs a new `Number` whose form is `3`; `txn.a * txn.b` above constructs a new `Number` whose form is `OpRef.Get("$a/mul", URI("b"))`. When debugging, it can be helpful to print the form of a `State` using the `form_of` function.

When using Python to develop a Tinychain service, it's important to remember that the output of your code is a Tinychain *compute graph* which will be served by a Tinychain host; your Python code itself won't be running in production. This means that you can't use Python control flow operators like `if` or `while` the way that you're used to. For example:

```python
@tc.get_op
def to_feet(txn, meters: tc.Number) -> tc.Number:
    # IMPORTANT! don't use Python's if statement! use tc.If!
    return tc.If(
        meters >= 0,
        meters * 3.28,
        tc.error.BadRequest("negative distance is not supported"))
```

For the same reason, it's important to use type annotations in your Tinychain Python code. Otherwise, you and your users (and the Tinychain client itself) won't know the return types of your methods! Without the type annotation in `meters: tc.Number` above, the argument `meters` would just be a `Value` and would not support operators like `*` or `>`.

It's also important to keep in mind that Tinychain by default resolves all dependencies concurrently, and does not resolve unused dependencies. Consider this function:

```python
@tc.post_op
def num_rows(txn):
    max_len = 100
    schema = tc.schema.Table(
        [tc.Column("user_id", tc.Number)],
        [tc.Column("name", tc.String, max_len), tc.Column("email", tc.String, max_len)])

    txn.table = tc.Table((key, value))
    txn.table.insert((123,), ("Bob", "bob.roberts@example.com"))
    return txn.table.count()
```

This Op will *always* resolve to *zero*. This may seem counterintuitive at first, because you can obviously see the `table.insert` statement, but notice that the return value `table.count` does not actually depend on `table.insert`; `table.insert` is only intended to create a side-effect, so its result is unused. To handle situations like this, use the `After` flow control:

```python
@tc.post_op
def num_rows(txn):
    max_len = 100
    schema = tc.schema.Table(
        [tc.Column("user_id", tc.Number)],
        [tc.Column("name", tc.String, max_len), tc.Column("email", tc.String, max_len)])

    txn.table = tc.Table(schema)
    return tc.After(
        txn.table.insert((123,), ("Bob", "bob.roberts@example.com")),
        txn.table.count())
```

Now, since the program explicitly indicates that `table.count` depends on a side-effect of `table.insert`, Tinychain won't execute `table.count` until after the call to `table.insert` has completed successfully.

## Object orientation

One of Tinychain's most powerful features is its object-oriented API. You can use this to define your own classes, which must inherit from exactly one class, which ultimately inherits from a native class. For example:

```python
from __future__ import annotations # needed until Python 3.10

LINK = "http://example.com/app/area" # <-- edit this

class Distance(tc.Number):
    __uri__ = tc.URI(LINK) + "/Distance"

    @tc.get_method
    def to_feet(self) -> Feet:
        return tc.error.NotImplemented("abstract")

    @tc.get_method
    def to_meters(self) -> Meters:
        return tc.error.NotImplemented("abstract")

class Feet(Distance):
    __uri__ = tc.URI(LINK) + "/Feet"

    @tc.get_method
    def to_feet(self) -> Feet:
        return self

    @tc.get_method
    def to_meters(self) -> Meters:
        return self / 3.28

class Meters(Distance):
    __uri__ = tc.URI(LINK) + "/Meters"

    @tc.get_method
    def to_feet(self) -> Feet:
        return self * 3.28

    @tc.get_method
    def to_meters(self) -> Meters:
        return self
```

Note that Tinychain does not have any concept of member visibility, like a "public" or "private" method. This is because Tinychain objects are meant to be sent over the network and used by client code, making a "private" method meaningless (and deceptive to the developer implementing it). If you want to hide an implementation detail from the public API of your class, use a Python function outside your class definition.

## Chain: persistent mutable state

In order to serve a dynamic application, you'll have to have a way of updating your service's persistent state. To do this you can use a `Chain`, a data structure which keeps track of mutations to a `Collection` or `Value` in order to maintain the consistency of that `State` across every replica of a `Cluster` (see below for details on `Cluster`).


```python
# from the example in test_replication.py

class Rev(tc.Cluster):
    __uri__ = tc.URI(f"http://127.0.0.1:8702/app/test/replication")

    def _configure(self):
        # this cluster has a SyncChain called "rev"
        self.rev = tc.Chain.Sync(0)

    # this method looks up the value of "rev" within a transaction
    @tc.get_method
    def version(self) -> tc.Number:
        return self.rev

    # this method updates the value of "rev" within a transaction
    @tc.post_method
    def bump(self, txn):
        # note: the value of "version()" must be assigned a name
        # in order to be addressable by the instance method __add__ below
        txn.rev = self.version()
        return self.rev.set(txn.rev + 1)
```

When you start a Tinychain host with a cluster definition, it will assume that there is a hosted cluster with the given URI ("http://.../app/test/replication" in the example above) and attempt to join that cluster as a replica. If the cluster URI has no host address, or is the same as the address of the running host, the host will serve a single replica of a new cluster. Watch out for versioning issues! In production, it's best to end your cluster URI with a version number which you can update in order to release a new, non-backwards-compatible version with different data and methods.

## Cluster: hosting your service

Of course, in order for your service to actually be useful to users, you have to put it online! You can do this using the same object-oriented API that you used to build the service:

```python
TC_PATH = "/path/to/tinychain/binary" # <-- edit this

# optionally, edit these also
CONFIG_PATH = "~/config/my_service.json"
WORKSPACE = "/tmp/tc/workspace"
DATA_DIR = "/tmp/tc/data"

# define the service
class AreaService(tc.Cluster):
    __uri__ = tc.URI(LINK)

    def _configure(self):
        self.Distance = Distance
        self.Feet = Feet
        self.Meters = Meters

    @tc.post_method
    def area(self, txn, length: Distance, width: Distance) -> tc.Number:
        txn.length_m = length.to_meters()
        txn.width_m = width.to_meters()
        return txn.length_m * txn.width_m

if __name__ == "__main__":
    # write the definition to disk
    tc.write_cluster(CONFIG_PATH)

    # start a new Tinychain host to serve AreaService
    host = tc.host.Local(TC_PATH, WORKSPACE, DATA_DIR, [CONFIG_PATH], force_create=True)

    # verify that it works as expected
    service = tc.use(AreaService)
    params = {"length": service.Meters(5), "width": service.Meters(2)}
    assert self.host.post("/app/area/area", params) == 10
```

You can see more in-depth examples in the [tests](http://github.com/haydnv/tinychain/tree/master/tests) directory.

## Calling another service from your own

Arguably the most powerful feature of Tinychain's Python client is the ability to interact with other services over the network like any other software library, using the same code that defines the service. This eliminates a huge amount of synchronization, validation, and conversion code relative to older microservice design patterns, as well as the need to write separate client and server libraries (although you're still free to do this for security purposes if you want). For example, if a client needs to call `AreaService`, they can use the exact same Python class that defines the service itself:

```python
from area import AreaService

class ClientService(tc.Cluster):
    __uri__ = tc.URI("http://127.0.0.1:8702/app/clientservice")

    @tc.get_method
    def room_area(self, txn, dimensions: tc.Tuple) -> Meters:
        area_service = tc.use(AreaService)
        txn.length = area_service.Meters(dimensions[0])
        txn.width = area_service.Meters(dimensions[1])
        return area_service.area(length=txn.length, width=txn.width)
```
