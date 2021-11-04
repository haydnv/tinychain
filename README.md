# TINYCHAIN

[![Crates.io][crates-badge]][crates-url]
[![Docs.rs][docs-badge]][docs-url]
[![Gitter][gitter-badge]][gitter-url]

[crates-badge]: https://img.shields.io/crates/v/tinychain.svg
[crates-url]: https://crates.io/crates/tinychain
[docs-badge]: https://docs.rs/tinychain/badge.svg
[docs-url]: https://docs.rs/tinychain/
[gitter-badge]: https://badges.gitter.im/tinychain-host/community.svg
[gitter-url]: https://gitter.im/tinychain-host/community

TinyChain is an all-in-one database + application server with support for blockchains, graphs, tables, and tensors.

TinyChain features automatic, out-of-the-box support for cross-service transactions--meaning that it brings the transactionality of a traditional database up to, and across, the application layer. TinyChain also features automatic multithreading and GPU acceleration. These and other features are detailed in the [key features](#key-features) section.

TinyChain is early beta software. Many features are not fully tested, or not yet available in the public API. You should not assume that TinyChain is secure, and you should make regular backups of your data.

If you're not sure whether TinyChain is for you, or if you have a question or find a bug, please [start a discussion](https://github.com/haydnv/tinychain/discussions)!

## What it does

### For developers

TinyChain is an all-in-one backend host which allows you to rapidly prototype a complex application and scale it without rewriting. You can easily split a monolithic application into many microservices, or combine multiple services into one monolithic application. TinyChain optimizes your application's performance with automatic concurrency (multithreading) and GPU acceleration, with no extra code required. There is a Python client provided (try it out with `pip3 install tinychain`) but you can easily build your own client in any language by implementing TinyChain's JSON data description and protocol (see [technical details](#technical-details)). You can find more information in the [data structures](#data-structures) section, the [client README](https://github.com/haydnv/tinychain/tree/master/client), the [code examples](https://github.com/haydnv/tinychain/tree/master/tests), and the [tutorial videos](https://www.youtube.com/channel/UCC6brO3L3JR0wUiMSDoGjrw).

### For operations/DevOps

TinyChain eliminates the need to manage an ever-growing "stack" of platform software which sometimes requires mutually-incompatible dependencies. With a TinyChain application, you no longer need tools like Docker and Kubernetes to package and deploy your application backend, because its "stack" is simply the TinyChain host software. You can easily deploy your application to a different cloud provider, or on-premises for a client, simply by starting up a new TinyChain host.

### For data scientists

With TinyChain, data scientists can easily analyze live replicas of a production database, eliminating the need to copy entire tables and databases out of the platform which tracks and enforces ownership of the data. This also eliminates the need to maintain a separate platform (e.g. TensorFlow Serving) in order to serve models. TinyChain also allows the construction of stateful models, e.g. a recurrent neural network (RNN)\* with a per-user state in memory, or a model which updates a database when it encounters an unexpected input.

### For product owners & executives

TinyChain is designed with many unique features to minimize the operational risk of hosting your customers' data. For example, TinyChain is the only database which supports *hypothetical queries*, which allow developers to examine the real consequences of potentially-destructive database updates without actually applying the updates. TinyChain is also the only database, and the only blockchain platform, designed from the ground up for compliance with data privacy laws like [GDPR](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) and [CCPA](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act).

TinyChain's all-in-one approach can also reduce operational costs by removing the need for developers to maintain a broad familiarity with a wide variety of specific platform tools, and removing the operational separation between application development and data science: TinyChain is a single platform which is useful to both developers and data scientists.

### For end-users

End users won't see TinyChain directly, but it makes the customer experience better by making cloud services faster, more reliable, and more cost-efficient to operate, as well as lowering the risk of a data breach by eliminating the need to make copies of customer data for analysis.

## Getting started

If your organization already has a TinyChain host or cluster up and running, all you should need is the Python client. You can install it with `pip3 install tinychain` (this may be `pip install tinychain` on very new systems).

If you're completely new to TinyChain, you'll need to run a TinyChain host in order to try it out. This is easy to do with Docker:

```bash
# build the Dockerfile from the GitHub repo, then run a new container with TinyChain listening on host port 8702
# the "-it" option also opens an interactive terminal
docker run -d -it -p 8702:8702/tcp $(docker build https://github.com/haydnv/tinychain.git -q) ./tinychain --address=0.0.0.0
```

For more advanced applications, see INSTALL.md for detailed installation instructions.

## Key features

| | TinyChain | Django (Python) | Ethereum | MongoDB | MySQL/MariaDB | Neo4j | Oracle | Spanner | Spring + Hibernate (Java) | PostgreSQL | TensorFlow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| At-rest encryption | \* | | | x | x | x | x | x | | x | |
| Automatic concurrency/multithreading | x | | | x | x | x | x | x | | x | x |
| Automatic GPU acceleration | x | | x | | | | | | | | x |
| Automatic sharding | \* | | | x | | | x | x | | | |
| Blockchain compliant w/ GDPR & CCPA | \* | | | | | | | | | | |
| Built-in cryptocurrency | | | x | | | | | | | | |
| Cross-service transactions | x | | x | | | | | | | | |
| Database tables | x | | | x | x | | x | x | | x | |
| Document database | | | | x | | | x | | | | |
| Distributed applications (dapps) | x | | x | | | | | | | | |
| Graph database | x | | | | | x | x | | | | |
| Hypothetical queries | x | | | | | | | | | | |
| Large ML library | | | | | | | | | | | x |
| JSON RPC interface | x | x | x | x | | x | | | x | | |
| Media (audio, video, image) support | \* | | | | | | | | | | x |
| Millisecond latency | x | x | | x | x | x | x | x | x | x | x |
| Object-oriented API | x | x | | x | | | | | x | | x |
| Object-relational mapping (ORM) | \*\* | x | | x | | | | | x | | |
| Stateful ML models | \*\* | | | | | | | | | | | 
| Tensor computation | x | | | | | | | | | | x |
| Transactional reads & writes | x | | x | x | x | x | x | x | | x | |

### Safety

 * **Synchronization**: TinyChain's distributed consensus algorithm automatically enables cross-service transactions within any set of services which support the TinyChain protocol.
 * **Hypothetical queries**: making changes to a production database is dangerous, particularly when the changes are inherently destructive (like dropping a column from a database table). TinyChain's hypothetical query feature mitigates this risk by allowing the developer to interact with their service *as if* a potentially-destructive change had been applied, without actually committing the change.
 * **Data encapsulation**: TinyChain integrates the database layer with the application layer of a cloud service, meaning that data never has to leave the application that authorizes access to it. Also, because TinyChain is a self-contained cloud-native runtime, a TinyChain service can easily be deployed to a user to run on-premises, meaning that the user never has to grant the vendor access to their data.
 * **Memory safety**: Around [70% of serious security bugs](https://www.zdnet.com/article/microsoft-70-percent-of-all-security-bugs-are-memory-safety-issues/) are memory-safety bugs. TinyChain is written in [Rust][rust-lang], which mitigates this risk by building memory safety into the language itself.

### Performance

 * **Concurrency**: TinyChain ops are automatically executed concurrently, although a developer can use the provided flow control operators to modify this behavior.
 * **Hardware acceleration**: The Tensor and Graph data types support automatic hardware acceleration on CUDA and OpenCL backends, no extra code needed.
 * **Native speed**: The TinyChain runtime itself is written in [Rust][rust-lang], a systems programming language with native support for memory-safe multithreading and compute performance comparable to C. All user-defined TinyChain classes must extend a native class implemented in Rust so that basic operations will benefit from native speed.
 * **Transactional filesystem cache**: TinyChain's transactional filesystem layer has a built-in memory cache with least-frequently used (LFU) eviction. Every Chain and Collection is built on top of this transactional filesystem, providing the benefits of in-memory speed for small collections and automatic filesystem caching for large collections. This means that, for example, you can write the same code to process a 1MB tensor or a 1TB tensor, without worrying about chunking the data to load it into memory.

[rust-lang]:http://www.rust-lang.org/

### Development velocity

 * **Rapid prototyping**: Set up a REST API for your service in just a few minutes and scale the application to your entire user base without rewriting.
 * **Object orientation**: The [Python client](http://github.com/haydnv/tinychain/tree/master/client) allows defining a TinyChain service as a Python package, which can be then be distributed to other developers, who can use the same Python code to interact with your (hosted) service from their own, without writing any boilerplate integration code, or defining two different APIs for internal and external use.
 * **Service mesh**: The TinyChain runtime acts as a service mesh for your TinyChain services, allowing for cross-service debugging and analysis, among other features, no extra integration code needed.
 * **Portability**: A TinyChain service can run in any cloud environment, or be distributed across many heterogenous clients, no Docker or Kubernetes required. A TinyChain developer can also choose make some or all of their service's functionality availble for a client to run on-premises, so that the client never has to give the service vendor access to their customer data.

## Data structures

 * **Cluster**: a collection of **Chain**s and **Op**s responsible for managing consensus relative to other **Cluster**s on the network
 * **Chain**: A record of mutations applied to a subject **Collection** or **Value**
    * **SyncChain**: A **Chain** with one block, which contains only the data necessary to recover from a transaction failure (e.g. in the event of a power failure)
    * **BackupChain**\*: A **Chain** whose blocks are deleted once they reach a certain age, and replaced with a copy of the **Chain**'s subject at that time
    * **BlockChain**: A **Chain** with a record of every mutation in the history of its **Collection** or **Value**
    * **CompliantChain**\*: A **Chain** which retains all history by default, but which allows purging all data owned by a given (anonymous) user ID, for compliance with legal requirements like [GDPR](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) and [CCPA](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act)
    * **ReduceChain**\*: A **Chain** which defines a reduce method to compress old blocks, useful for metrics (e.g. to reduce per-second statistics to per-minute, per-minute to per-hour, etc.)
 * **Collection**
    * **BTree**: A [B-Tree](https://en.wikipedia.org/wiki/B-tree), used to index tabular data
    * **Table**: A database table, which supports one or more **BTree** indices
    * **Tensor**: An n-dimensional array of numbers which supports both sparse and dense representations, useful for machine learning applications
    * **Graph**: A graph database which uses a sparse **Tensor** to compute relationships between rows in its **Table**s
 * **Scalar**
    * **Value**: a generic value type such as a string or number which can be collated and stored in a **BTree**
    * **Ref**: a reference to another value which must be resolved as part of a transaction
    * **Op**: a user-defined executable function

## Technical details

This information is documented in case you want to develop your own general-purpose TinyChain client, or if you need to do an in-depth security or risk analysis. If you're just a regular developer using the Python client, you don't have to worry about any of this!

### Data description

TinyChain exposes a JSON API over HTTP, and treats a subset of JSON as a Turing-complete application language. For example, this is a function to convert meters into feet:

```json
{"/state/scalar/op/get": ["meters", [
    {"/state/scalar/ref/if": [
        {"$meters/gte": 0},
        {"$meters/mul": 3.28},
        {"/error/bad_request": "Negative distance is not supported"}
    ]}
]]}
```

Obviously writing this by hand gets unwieldy very quickly, which is why a [Python client](https://github.com/haydnv/tinychain/tree/master/client) is provided. Here's the same function defined using the Python client:

```python
@tc.get_op
def to_feet(txn, meters: tc.Number) -> tc.Number:
    return tc.If(
        meters >= 0,
        meters * 3.28,
        tc.error.BadRequest("negative distance is not supported"))
```

You can look in the [tests](https://github.com/haydnv/tinychain/tree/master/tests) directory for more detailed examples.

### Replication

A TinyChain cluster uses a variation on [fast Byzantine multi-Paxos consensus](https://en.wikipedia.org/wiki/Paxos_(computer_science)#Message_flow:_Fast_Byzantine_Multi-Paxos,_steady_state).

The life of a replica *R* of a cluster *C* is:

1. Host loads cluster config
1. If any peer is hosting a cluster at the same path, a replication request is sent, and the state of *R* is updated to match the state of *C*; otherwise, *R* is assumed to be the latest state of *C*
1. *R* will handle transaction requests for *C* according to the flow below until its host is shut down or it attempts to commit a transaction and fails
1. If *R* attempts to commit a transaction and fails, it will stop accepting requests and attempt to re-join *C* by starting from step 1. above (assuming that automatic restarts are configured with systemd)

The flow of operations within a single transaction is:

1. A replica host *R* of cluster *C* with *N* total replicas receives a new transaction request *T*
1. *R* claims ownership of the transaction
1. For PUT and DELETE operations, the request *T* itself is replicated; for user-defined GET and POST requests, write operations which are part of *T* are replicated to all other hosts in *C*
1. Each dependent cluster *Cx* receives a request *Tx*, claims leadership of *Tx*, notifies the owner *C* of its participation, and replicates *Tx* to all hosts in *Cx*
1. If any host responds with error 409: conflict, the transaction is rolled back and error 409 is returned to the end-user
1. Otherwise, if at least (*N* / 2) + 1 hosts in each participating cluster respond with success, each cluster *C* removes the unsuccessful hosts from its replica set, commits the transaction, and responds to the end-user
1. Otherwise, the transaction is rolled back and an error is returned to the end-user

*Important note*: the TinyChain protocol does not support trustless replication. Do not allow untrusted replicas to join your cluster. A single malicious replica can significantly degrade performance, or even halt all updates entirely, by creating extra work to reach consensus; it can also report false information to your clients and the network as a whole.

### Authentication

TinyChain uses recursive JSON web tokens provided by the [rjwt](http://docs.rs/rjwt/) library. Each TinyChain host which handles part of a transaction must validate the auth token of the incoming request, sign it with its own private key, and forward it to any downstream dependencies. Note the unusual security consideration of a recursive token: a downstream dependency receives all of the upstream tokens, and therefore is authorized to take any action which an upstream dependency is authorized to take. For this reason, it's very important to use only a minimal scope to authenticate the end-user, and grant further scopes as narrowly as possible.

### Life of a transaction

Example:

```
|   user@example.com    |         retailer.com       |             factory.com          |          shipping.com        | bank.com
|-----------------------|----------------------------|----------------------------------|------------------------------|----------
| Buy 1 widget for $20 -> Debit user's account $20 -------------------------------------------------------------------->          
|                       | Make 1 widget -------------> Debit retailer's account $10 ----------------------------------->
|                       |                            | Ship 1 widget to user's address -> Debit retailer's account $5 ->          
```

* `user@example.com` initiates the transaction by sending a request to `retailer.com`, with an auth token signed by `example.com` whose scopes limit its use to the intended action
    * `retailer.com` claims ownership of the transaction by creating a new, signed request token which includes the original token
        * `retailer.com` sends a request to `bank.com` to debit $20 from the user's account
        * `retailer.com` sends a request to `factory.com` to manufacture 1 widget
            * `factory.com` sends a request to `bank.com` to debit $10 from `retailer.com`'s account
            * `factory.com` sends a request to `shipping.com` to schedule a shipment to the user's address, and charge `retailer.com`
                * `shipping.com` sends a request to `bank.com` to debit $5 from `retailer.com`'s account

Requests at the same indentation level above are executed concurrently, after validating the incoming request token. Acceptance of a request with transaction ID X forbids acceptance of any other request with transaction ID X, unless they share the same owner.

Note that this is only the case in a mutation (a write transaction). A read-only transaction does not need to perform any explicit synchronization with the transaction owner, only to lock the given transaction ID.

The concurrency control flow of this sequence of operations, starting from transaction number (X - 1) is:
1. Client initiates transaction X
1. Service endpoint validates auth token
1. Service endpoint claims transaction X, becoming the transaction owner
1. Service endpoint executes requested operation
    1. Dependency receives request
    1. Dependency validates auth token
    1. If the dependency is part of a different **Cluster** than the transaction owner, it notifies the owner of its participation (this is necessary to synchoronize dependencies further downstream)
    1. Dependency executes requested operation and replies to the transaction owner
1. Transaction owner notifies dependencies of success, updating their latest committed transaction to X
1. Transaction owner updates its latest committed transaction to X
1. Transaction owner replies to the client

This functionality is implemented automatically for any service using the TinyChain host software.

*Important note*: this cross-service consensus algorithm trusts each service to a) recover from a crash without losing state, and b) communicate that it has committed a transaction honestly and correctly. In other words, it is not a new and trustless protocol, but a new formalization of an existing ad-hoc procedure. An application which requires completely trustless transactions, like a distributed cryptocurrency, should use a single replicated **Cluster**.

\* Not yet implemented

\*\* Not yet available in the public API

