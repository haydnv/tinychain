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
