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

## The Tinychain Environment

Tinychain is both a database platform (like MySQL) and an analytics/ML platform (like TensorFlow)
as well as a Turing-complete runtime (like Ethereum) with blockchain-powered versioning features.**
In general, Tinychain development consists of constructing a deferred execution graph (this is
loosely modeled on TensorFlow v1), testing it, and saving it to a [Cluster](#cluster) for users to
execute. For the sake of brevity, these examples omit calling `POST /sbin/transact/execute` on
the computation graph.

The main components of Tinychain that you'll need to be familiar with are:

### Value

A **Value** such as an integer, Link, or Op is Tinychain's most primitive data type. A Value is
immutable and must always fit in its host's main memory. Values are accessible under
`/sbin/value/...`.

Example:

```
curl -G "http://127.0.0.1:8888/sbin/value/number/int/32" --data-urlencode 'key=42'
```

For more examples, check out the [test suite](https://github.com/haydnv/tctest/blob/master/test.py).

### Op

An **Op** is a type of Value. It deserves special mention because Ops power Tinychain's programming
and execution model. The [Python client](http://github.com/haydnv/tinychain.py) defines Tinychain's
built-in Ops in a (hopefully) fairly intuitive way:

```
import tinychain as tc
from tinychain.util import to_json

a = tc.I32("a", 2)
b = tc.I32("b", 3)

print(to_json(a))       # {'/sbin/value/number/int/32': [2]}
print(to_json(b))       # {'/sbin/value/number/int/32': [3]}
print(to_json(a + b))   # {'$a/add': [{'$b': []}]}
```

You can also define your own ops as part of a Chain in order to control allowable reads and
mutations. See the [Chain](#chain) section for examples.

### Collection
 * **BTree**: a B-Tree index. There may be situations where you want to create a B-Tree on-the-fly,
like within the context of an Op that needs to store an indefinite number of intermediate results,
but in general you should use a Table instead. You can instantiate a B-Tree by calling
`GET /sbin/collection/btree`.

 * **Table**: a SQL-style table. For reasons of security and expediency, Tinychain does not include a
SQL parser, but the Table interface is designed to be familiar to anyone who has used a SQL
database before.

Example:

```
import tinychain as tc

key = (("user_id", tc.U64),)
values = (("name", tc.UString), ("email", tc.UString))
# create a table with an index on the "email" column
table = tc.Table((key, values), [("email_index", ("email",))])
table.insert((123,), ("Alice", "alice@example.com"))
table.insert((234,), ("Bob", "bob@example.com"))
table.count() # 2
table.slice({"name": "Bob"}).select(("email",)) # [("bob@example.com",)]
```

 * **Tensor**: a regular multidimensional collection of numbers which comes in Dense and Sparse
flavors. Both support hardware acceleration, including via GPU, but using a DenseTensor will be
faster unless the data really is very sparse. You can instantiate a DenseTensor by calling
`GET /sbin/collection/tensor/dense` or a SparseTensor by calling
`GET /sbin/collection/tensor/sparse`.

Example**:

```
import tinychain as tc

tensor = tc.Tensor.create(tc.I32, [2, 10], value=1) # equivalent to numpy.ones([2, 10])
x = tensor.sum(axis = 0) # [10, 10]
```

 * **Graph**: a SparseTensor-based graph data store is planned for a future version of Tinychain

### Chain
 * **Null**: a Null Chain is a Chain which can define methods but does not save any persistent
record of mutation requests. This can be useful in situations where a resource is frequently
updated, but there is no need to keep track of those updates (for example, a count of page views on
a blog).
 * **Block****: a Block Chain is a Chain which gates access to a mutable resource and saves a
persistent record of all mutation requests. This is useful in situations where auditability is
critically important, like employee requests to access customer data.
 * **Compliant****: a Compliant Chain is similar to a Block Chain except that it allows purging all
data owned by a given user in order to comply with legal requirements like
[GDPR](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation),
[CCPA](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act), etc.
 * **Aggregate****: an Aggregate Chain is a chain which aggregates its own contents in order to
avoid wasting storage space. This is useful in situations where you need to perform real-time
statistical analysis but don't want to keep fine-grained data (like a log of API requests) forever.

Example**:

```
import tinychain as tc


class ExpenseAccount(tc.ValueBlockChain):
    @classmethod
    def create(cls):
        return super(tc.ValueBlockChain, cls).create(tc.I32, 0.)

    @tc.put
    def deposit(self, amount: tc.I32):
        balance = self.value()
        return self.value(balance + amount)

    @tc.post
    def withdraw(self, amount: tc.I32):
        balance = self.value()
        return tc.If(amount < balance,
            self.value(balance - amount),
            tc.BadRequestError("Insufficient funds"))
```

### Cluster**

A Cluster is, from the developer's perspective, the primary synchronization mechanism of Tinychain.
Analogous to a database in MySQL, a Tinychain Cluster is a collection of Chained states which
supports concurrent atomic transactional access by multiple users. All persistent data in Tinychain
must be served by a Cluster.

Example**:

```
import tinychain as tc


class InventoryChain(tc.BlockChain):
    @classmethod
    def create(cls):
        key = (("sku", tc.U64),)
        values = (("name", tc.UString), ("quantity", tc.U32))
        return super(tc.BlockChain, cls).create(tc.Table, (key, values))


class VendorDatabase(tc.Cluster):
    def __init__(self, path):
        self.inventory = InventoryChain
        tc.Cluster.__init__(self, path)


if __name__ == "__main__":
    cluster = CustomCluster("/app/vendor")
    host = tc.LocalHost(PATH, "cluster_replica1", 8888, hosted=["/app/vendor"])
    host.replicate(cluster)

    inventory = InventoryChain("inventory", tc.Link("/app/vendor/inventory")).select(("quantity",)
    num_items = tc.Tensor.create_from(inventory).sum()
    print(host.execute(num_items))
```

Sharding support is planned for a future version of Tinychain.

### Transaction

A **Transaction** is an atomic operation, which can include multiple sub-operations. Tinychain uses
transactions to present a consistent global state across the entire network of all Tinychain hosts.
A Cluster will resolve transactions using the
[Paxos](https://en.wikipedia.org/wiki/Paxos_%28computer_science%29) consensus algorithm using each
member Chain as a [write-ahead log](https://en.wikipedia.org/wiki/Write-ahead_logging).


** This feature is not yet implemented

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
