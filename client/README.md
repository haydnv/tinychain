
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

@tc.get_op
def hello():
    return tc.String("Hello, World!")

if __name__ == "__main__":
   host = tc.host.Local(TC_PATH, WORKSPACE)
   print(host.post(ENDPOINT, hello))
```

## Overview

The Tinychain client provides a developer-friendly API to build Tinychain transactions (i.e., distributed compute graphs) using the familiar Python language, but without any of the associated restrictions on performance or locality. A Tinychain transaction can span any number of networked hosts and automatically takes advantage of the hosts' concurrent and parallel computing resources (like multiple cores and GPU acceleration) without any extra application code. A Tinychain service defined using the Python client can be packaged using familiar distribution mechanisms like [Pip](http://pypi.org/project/pip) so that other Tinychain developers can use it just like a regular Python developer would use a regular Python library. This can save your clients a lot of time and hassle when integrating your cloud API into their service.

Every value in a Tinychain client app is either a `State`, representing a Tinychain state like a `Number` or a `Chain`, or a `Ref`, which tells a Tinychain transaction how to access or calculate a particular `State`. For example:

```python
@tc.get_op
def example(txn) -> tc.Number:
    txn.a = tc.Number(5) # this is a State
    txn.b = tc.Number(10) # this is a State
    txn.product = txn.a * txn.b # this is a Ref
    return txn.product
```

The constructor of a `State` always takes exactly one argument, which is the form of that `State`. For example, `tc.Number(3)` constructs a new `Number` whose form is `3`; `txn.a * txn.b` above constructs a new `Number` whose form is `OpRef.Get("$a/mul", IdRef("b"))`. When debugging, it can be helpful to print the form of a `State` using the `form_of` function.

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

For the same reason, it's important to use type annotations in your Tinychain Python code. Otherwise, you and your users (and the Tinychain client itself) won't know the return types of your methods! Without the type annotation in `meters: tc.Number` above, the argument `meters` would be of type `IdRef` and would not support operators like `*` or `>`.

It's also important to keep in mind that Tinychain by default resolves all dependencies concurrently, and does not resolve unused dependencies. Consider this function:

```python
@tc.post_op
def num_rows(txn):
    key = [("user_id", tc.Number)]
    value = [("name", tc.String), ("email", tc.String)]

    txn.table = tc.Table(key + value)
    txn.table.insert((123, "Bob", "bob.roberts@example.com"))
    return txn.table.count()
```

The output of this function will always be ZERO. This may seem counterintuitive at first, because you can obviously see the `table.insert` statement, but notice that the return value `table.count` does not actually depend on `table.insert`; `table.insert` is only intended to create a side-effect, so its return value is unused. To handle situations like this, use the `After` flow control:

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
