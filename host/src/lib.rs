//! TinyChain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! TinyChain protocol.
//!
//! TinyChain currently supports `BlockChain`, `BTree`, `Table`, and `Tensor` collection types,
//! with more planned for the future.
//!
//! TinyChain is intended to be used as an executable binary (i.e., with `cargo install`) via its
//! HTTP API. For usage instructions and more details, visit the repository page at
//! [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain).

pub mod http;

/// The minimum size of the transactional filesystem cache, in bytes
pub const MIN_CACHE_SIZE: usize = 5000;
