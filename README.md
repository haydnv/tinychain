# TINYCHAIN

Tinychain is an all-in-one database + application server with support for tensors, tables, and blockchains.

Tinychain is still in early alpha status and is not ready for production use. Many core features are only partially implemented, or not yet available in the public API.

## Getting started

The easiest way to get started is to download the latest release from GitHub here: https://github.com/haydnv/tinychain/releases

Tinychain is currently only available for 64-bit Linux (you should be able to build from source on other architectures, but this has not been tested).

Try this "Hello, World!" program:

```bash
curl -G "http://localhost:8702/state/scalar/value/string" --data-urlencode 'key="Hello, world!"'
```

## The Tinychain protocol

Tinychain exposes a JSON API over HTTP, and treats a subset of JSON as a Turing-complete application language. For example, this is a function to convert meters into feet:

```json
{"/state/scalar/op/get": ["meters", [
    {"/state/scalar/ref/if": [
        {"$meters/gte": 0},
        {"$meters/mul": 3.28},
        {"/error/bad_request": "Negative distance is not supported"}
    ]}
]]}
```

Obviously writing this by hand gets unwieldy very quickly, which is why a [Python client](https://github.com/haydnv/tinychain/tree/master/client) is provided. You can look in the [tests](https://github.com/haydnv/tinychain/tree/master/client/tests) directory for more detailed examples.

## Key features

 * **Synchronization**: Tinychain's distributed Paxos algorithm automatically enables cross-servide transactions across any set of services which support the Tinychain protocol (which, again, is basically just JSON).

More documentation is coming soon.

