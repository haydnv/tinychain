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

Tinychain is an all-in-one database + application server with support for blockchains, graphs, tables, and tensors.

Tinychain is useful in any situation which requires developing and deploying a cloud service. Anytime you consider developing a service with, for example, MySQL or MongoDB or Neo4j and Apache or Nginx or Django, you can probably get the same thing done faster and more scalably with Tinychain. The major advantage of Tinychain is its automatic, out-of-the-box support for cross-service transactions--meaning that it brings the transactionality of a traditional database up to, and across, the application layer.

Tinychain is early beta software. Many features are only partially implemented, or not fully tested, or not yet available in the public API. You should not assume that Tinychain is secure, and you should make regular backups of your data.

If you're not sure how to get started, or have a question, or find a bug, please [start a discussion](https://github.com/haydnv/tinychain/discussions)!

## Getting started

The easiest way to get started is to download the latest release from GitHub here: https://github.com/haydnv/tinychain/releases. Binary releases are currently only available for 64-bit Intel Linux (you should be able to build from source on other architectures, but this has not been tested).

If you use the Rust programming language, you can install Tinychain on any platform by running `cargo install tinychain`.

Try this "Hello, World!" program:

```bash
$ ./tinychain &
# ...
$ curl -G "http://localhost:8702/state/scalar/value/string" --data-urlencode 'key="Hello, world!"'
```

There are instructions to set up a Cluster in the [client README]:(https://github.com/haydnv/tinychain/tree/master/client).

You can find more in-depth examples in the [tests](https://github.com/haydnv/tinychain/tree/master/tests) directory. There is also a series of tutorial videos on YouTube: [https://www.youtube.com/channel/UCC6brO3L3JR0wUiMSDoGjrw](https://www.youtube.com/channel/UCC6brO3L3JR0wUiMSDoGjrw)

## Key features

### Safety

 * **Synchronization**: Tinychain's distributed consensus algorithm automatically enables cross-service transactions within any set of services which support the Tinychain protocol.
 * **Hypothetical queries**: making changes to a production database is dangerous, particularly when the changes are inherently destructive (like dropping a column from a database table). Tinychain's hypothetical query feature mitigates this risk by allowing the developer to interact with their service *as if* a potentially-destructive change had been applied, without actually committing the change.
 * **Data encapsulation**: Tinychain integrates the database layer with the application layer of a cloud service, meaning that data never has to leave the application that authorizes access to it. Also, because Tinychain is a self-contained cloud-native runtime, a Tinychain service can easily be deployed to a user to run on-premises, meaning that the user never has to grant the vendor access to their data.
 * **Memory safety**: Around [70% of serious security bugs](https://www.zdnet.com/article/microsoft-70-percent-of-all-security-bugs-are-memory-safety-issues/) are memory-safety bugs. Tinychain is written in [Rust][rust-lang], which mitigates this risk by building memory safety into the language itself.

### Performance

 * **Concurrency**: Tinychain ops are automatically executed concurrently, although a developer can use the provided flow control operators to modify this behavior.
 * **Hardware acceleration**: The Tensor and Graph data types support automatic hardware acceleration on CUDA and OpenCL backends, no extra code needed.
 * **Native speed**: The Tinychain runtime itself is written in [Rust][rust-lang], a systems programming language with native support for memory-safe multithreading and compute performance comparable to C. All user-defined Tinychain classes must extend a native class implemented in Rust so that basic operations will benefit from native speed.
 * **Transactional filesystem cache**: Tinychain's transactional filesystem layer has a built-in memory cache with least-frequently used (LFU) eviction. Every Chain and Collection is built on top of this transactional filesystem, providing the benefits of in-memory speed for small collections and automatic filesystem caching for large collections. This means that, for example, you can write the same code to process a 1MB tensor or a 1TB tensor, without worrying about chunking the data to load it into memory.

[rust-lang]:http://www.rust-lang.org/

### Development velocity

 * **Rapid prototyping**: Set up a REST API for your service in just a few minutes and scale the application to your entire user base without rewriting.
 * **Object orientation**: The [Python client](http://github.com/haydnv/tinychain/tree/master/client) allows defining a Tinychain service as a Python package, which can be then be distributed to other developers, who can use the same Python code to interact with your (hosted) service from their own, without writing any boilerplate integration code, or defining two different APIs for internal and external use.
 * **Service mesh**: The Tinychain runtime acts as a service mesh for your Tinychain services, allowing for cross-service debugging and analysis, among other features, no extra integration code needed.
 * **Portability**: A Tinychain service can run in any cloud environment, or be distributed across many heterogenous clients, no Docker or Kubernetes required. A Tinychain developer can also choose make some or all of their service's functionality availble for a client to run on-premises, so that the client never has to give the service vendor access to their customer data.

## Data structures

 * **Cluster**: a collection of **Chain**s and **Op**s responsible for managing consensus relative to other **Cluster**s on the network
 * **Chain**: A record of mutations applied to a subject **Collection**\*\* or **Value**
    * **SyncChain**: A **Chain** with one block, which contains the data necessary to recover from a transaction failure (e.g. if the host crashes)
    * **BackupChain**\*: A **Chain** whose blocks are deleted once they reach a certain age, and replaced with a copy of the **Chain**'s subject at that time
    * **BlockChain**\*: A **Chain** with a record of every mutation in the history of its **Collection**\*\* or **Value**
    * **CompliantChain**\*: A **Chain** which retains all history by default, but which allows purging all data owned by a given (anonymous) user ID, for compliance with legal requirements like [GDPR](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) and [CCPA](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act)
    * **ReduceChain**\*: A **Chain** which defines a reduce method to compress old blocks, useful for metrics (e.g. to reduce per-second statistics to per-minute, per-minute to per-hour, etc.)
 * **Collection**\*\*
    * **BTree**\*\*: A [B-Tree](https://en.wikipedia.org/wiki/B-tree), used to index tabular data
    * **Table**\*\*: A database table, which supports one or more **BTree**\*\* indices
    * **Tensor**\*\*: An n-dimensional array of numbers which supports both sparse and dense representations, useful for machine learning applications
    * **Graph**\*: A graph database which uses a sparse **Tensor**\*\* to compute relationships between rows in its **Table**\*\*s
 * **Scalar**
    * **Value**: a generic value type such as a string or number which can be collated and stored in a **BTree**
    * **Ref**: a reference to another value which must be resolved as part of a transaction
    * **Op**: a user-defined executable function

## The Tinychain protocol

### Data description

Tinychain exposes a JSON API over HTTP, and treats a subset of JSON as a Turing-complete application language. For example, this is a function to convert meters into feet:

```json
{"/state/scalar/op/get": ["to_feet", [
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

A Tinychain cluster uses a variation on [fast Byzantine multi-Paxos consensus](https://en.wikipedia.org/wiki/Paxos_(computer_science)#Message_flow:_Fast_Byzantine_Multi-Paxos,_steady_state).

The life of a replica *R* of a cluster *C* is:

1. Host loads cluster config
1. If any peer is hosting a cluster at the same path, a replication request is sent, and the state of *R* is updated to match the state of *C*; otherwise, *R* is assumed to be the latest state of *C*
1. *R* will handle transaction requests for *C* according to the flow below until its host is shut down or it attempts to commit a transaction and fails
1. If *R* attempts to commit a transaction and fails, it will stop accepting requests and attempt to re-join *C* by starting from step 1. above (assuming that automatic restarts are configured with systemd)

The flow of operations within a single transaction is:

1. A replica *R* of cluster *C* with *N* total replicas receives a new transaction request *T*
1. *R* claims ownership of the transaction
1. For every write operation which is part of *T*, *R* notifies all other replica hosts
1. If any replica responds with error 409: conflict, the transaction is rolled back and error 409 is returned to the host.
1. Otherwise, if at least (*N* / 2) + 1 replicas respond with success, *R* removes the unsuccessful replicas from *C*, commits the transaction, and responds to the end-user
1. Otherwise, the transaction is rolled back and an error is returned to the end-user

*Important note*: the Tinychain protocol does not support trustless replication. Do not allow untrusted replicas to join your cluster. A single malicious replica can significantly degrade performance, or even halt all updates entirely, by creating extra work to reach consensus; it can also lie to your clients and introduce false information into the network as a whole.

### Authentication

Tinychain uses recursive JSON web tokens provided by the [rjwt](http://docs.rs/rjwt/) library. Each Tinychain host which handles part of a transaction must validate the auth token of the incoming request, sign it with its own private key, and forward it to any downstream dependencies.

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

This functionality is implemented automatically for any service using the Tinychain host software.

*Important note*: this cross-service consensus algorithm trusts each service to a) recover from a crash without losing state, and b) communicate that it has committed a transaction honestly and correctly. In other words, it is not a new and trustless protocol, but a new formalization of an existing ad-hoc procedure. An application which requires completely trustless transactions, like a distributed cryptocurrency, should use a single replicated **Cluster**.

\* Not yet implemented

\*\* Not yet available in the public API

