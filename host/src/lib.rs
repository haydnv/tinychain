//! Tinychain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! Tinychain protocol. Tinychain itself is also a Turing-complete application platform.
//!
//! Tinychain currently support `BlockChain`, `BTree`, and `Table` collection types,
//! with more planned for the future.
//!
//! Tinychain is intended to be used as an executable binary (i.e., with `cargo install`) via its
//! HTTP API. For usage instructions and more details, visit the repository page at
//! [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain).

use std::path::PathBuf;

pub use kernel::*;
pub use tc_btree as btree;
pub use tc_error as error;
pub use tc_table as table;
pub use tc_transact as transact;
pub use tc_value as value;
pub use tcgeneric as generic;

mod fs;
mod http;
mod route;

pub mod chain;
pub mod cluster;
pub mod collection;
pub mod gateway;
pub mod kernel;
pub mod object;
pub mod scalar;
pub mod state;
pub mod txn;

/// Initialize the transactional filesystem layer.
pub async fn mount(
    workspace: PathBuf,
    data_dir: Option<PathBuf>,
    cache_size: u64,
) -> tc_error::TCResult<(fs::Dir, Option<fs::Dir>)> {
    let cache = fs::Cache::new(cache_size);

    let workspace = fs::load(cache.clone(), workspace).await?;
    let data_dir = if let Some(data_dir) = data_dir {
        Some(fs::load(cache, data_dir).await?)
    } else {
        None
    };

    Ok((workspace, data_dir))
}
