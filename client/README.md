
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
    key = [("user_id", tc.Number)]
    value = [("name", tc.String), ("email", tc.String)]

    txn.table = tc.Table((key, value))
    txn.table.insert((123,), ("Bob", "bob.roberts@example.com"))
    return txn.table.count()
```

This Op will *always* resolve to *zero*. This may seem counterintuitive at first, because you can obviously see the `table.insert` statement, but notice that the return value `table.count` does not actually depend on `table.insert`; `table.insert` is only intended to create a side-effect, so its result is unused. To handle situations like this, use the `After` flow control:

```python
@tc.post_op
def num_rows(txn):
    key = [("user_id", tc.Number)]
    value = [("name", tc.String), ("email", tc.String)]

    txn.table = tc.Table(key + value)
    return tc.After(
        txn.table.insert((123, "Bob", "bob.roberts@example.com")),
        txn.table.count())
```

Now, since the program explicitly indicates that `table.count` depends on a side-effect of `table.insert`, Tinychain won't execute `table.count` until after the call to `table.insert` has completed successfully.

## Object orientation

One of Tinychain's most powerful features is its object-oriented API. You can use this to define your own classes, which must inherit from exactly one class, which ultimately inherits from a native class. For example:

```python
from __future__ import annotations # needed until Python 3.10

LINK = "http://example.com/app/myservice" # <-- edit this

class Distance(tc.Number, metaclass=tc.Meta):
    # make sure this is the URI which serves this class definition
    __uri__ = tc.URI(LINK) + "/distance"

    @tc.get_method
    def to_feet(self):
        return tc.error.NotImplemented("abstract")

    @tc.get_method
    def to_meters(self):
        return tc.error.NotImplemented("abstract")

class Feet(Distance):
    __uri__ = tc.URI(LINK) + "/feet"

    @tc.get_method
    def to_feet(self) -> Feet:
        return self

    @tc.get_method
    def to_meters(self) -> Meters:
        return self / 3.28

class Meters(Distance):
    __uri__ = tc.URI(LINK) + "/meters"

    @tc.get_method
    def to_feet(self) -> Feet:
        return self * 3.28

    @tc.get_method
    def to_meters(self) -> Meters:
        return self
```

Note that Tinychain does not have any concept of member visibility, like a "public" or "private" method. This is because Tinychain objects are meant to be sent over the network and used by client code, making a "private" method meaningless (and deceptive to the developer implementing it). If you want to hide an implementation detail from the public API of your class, use a Python function outside your class definition.

## Cluster: hosting your service

Of course, in order for your service to actually be useful to users, you have to put it online! You can do this using the same object-oriented API that you used to build the service:

```python
TC_PATH = "/path/to/tinychain/binary" # <-- edit this

# optionally, edit these also
CONFIG_PATH = "~/config/my_service.json"
WORKSPACE = "/tmp/tc/workspace"
DATA_DIR = "/tmp/tc/data"

# define the service
class MyService(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI(URL)

    def configure(self):
        # if your clients need to access your class definitions,
        # make sure to list them here so that Tinychain will make them available
        # via GET request

        self.Distance = Distance
        self.Feet = Feet
        self.Meters = Meters

    @tc.post_method
    def area(self, txn, length: Distance, width: Distance) -> Meters:
        return length.to_meters() * width.to_meters()

if __name__ == "__main__":
    # write the definition to disk
    tc.write_cluster(CONFIG_PATH)

    # start a new Tinychain host to serve MyService
    host = tc.host.Local(TC_PATH, WORKSPACE, DATA_DIR, [CONFIG_PATH], force_create=True)
    area = host.post("http://127.0.0.1:8702/app/myservice/area", length=5, width=10)
    assert area == 50
```

You can see more in-depth examples in the [tests](http://github.com/haydnv/tinychain/tree/master/tests) directory.

## Calling another service from your own

Arguably the most powerful feature of Tinychain's Python client is the ability to interact with other services over the network like any other software library, using the same code that defines the service. This eliminates a huge amount of synchronization, validation, and conversion code relative to older microservice design patterns, as well as the need to write separate client and server libraries (although you're still free to do this for security purposes if you want). For example, if a client needs to call `MyService`, they can use the exact same Python class that defines the service itself:

```python
from myservice import MyService, Distance, Meters

class ClientService(tc.Cluster, metaclass=tc.Meta):
    __uri__ = "http://clientwebsite.com/app/clientservice" # <-- edit this

    @tc.get_method
    def room_area(self, txn, dimensions: Tuple) -> Meters:
        myservice = tc.use(MyService) # note: MyService is running on a *different host*
        return myservice.area(length=dimensions[0], width=dimensions[1])
```

