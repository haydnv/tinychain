//! Tinychain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! Tinychain protocol. Tinychain itself is also a Turing-complete application platform.
//!
//! Tinychain is intended to be used as an executable binary (i.e., with `cargo install`) via its
//! HTTP API. For usage instructions and more details, visit the repository page at
//! [github.com/haydnv/tinychain](http://github.com/tinychain).

use std::path::PathBuf;

mod fs;
mod http;
mod route;

pub mod chain;
pub mod cluster;
pub mod gateway;
pub mod kernel;
pub mod object;
pub mod scalar;
pub mod state;
pub mod txn;

pub use kernel::*;
pub use tc_error;
pub use tc_transact;
pub use tc_value;
pub use tcgeneric;

/// Initialize the transactional filesystem layer.
pub async fn mount(
    workspace: PathBuf,
    data_dir: Option<PathBuf>,
    cache_size: usize,
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
