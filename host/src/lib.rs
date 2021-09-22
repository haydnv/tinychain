//! TinyChain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! TinyChain protocol. TinyChain itself is also a Turing-complete application platform.
//!
//! TinyChain currently supports `BlockChain`, `BTree`, `Table`, and `Tensor` collection types,
//! with more planned for the future.
//!
//! TinyChain is intended to be used as an executable binary (i.e., with `cargo install`) via its
//! HTTP API. For usage instructions and more details, visit the repository page at
//! [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain).

use std::path::PathBuf;

use futures::TryFutureExt;

pub use kernel::*;
pub use tc_btree as btree;
pub use tc_error as error;
pub use tc_table as table;
#[cfg(feature = "tensor")]
pub use tc_tensor as tensor;
pub use tc_transact as transact;
pub use tc_value as value;
pub use tcgeneric as generic;

mod fs;
mod http;
mod route;

pub mod chain;
pub mod closure;
pub mod cluster;
pub mod collection;
pub mod gateway;
pub mod kernel;
pub mod object;
pub mod scalar;
pub mod state;
pub mod stream;
pub mod txn;

const MIN_CACHE_SIZE: u64 = 5000;
