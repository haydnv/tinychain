//! Tinychain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! Tinychain protocol. Tinychain itself is also a Turing-complete application platform.
//!
//! For more details on the Tinychain project, visit the repository page at
//! [github.com/haydnv/tinychain](http://github.com/tinychain).
//!
//! This library is provided in the hope that it will be useful, but Tinychain is primarily designed
//! to be used via the HTTP API. The best way for new users to get started is to download the
//! latest binary release from [GitHub](http://github.com/haydnv/tinychain) and go through the
//! quickstart guide.
//!
//! This is an early alpha release of Tinychain. Many features are incomplete, unstable, or simply
//! not yet implemented. Tinychain is not ready for production use.

use std::path::PathBuf;

pub use generic;
pub use value;

pub use error;
pub use kernel::*;

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

/// Initialize the transactional filesystem layer.
pub async fn mount(
    workspace: PathBuf,
    data_dir: Option<PathBuf>,
    cache_size: usize,
) -> error::TCResult<(fs::Dir, Option<fs::Dir>)> {
    let cache = fs::Cache::new(cache_size);

    let workspace = fs::load(cache.clone(), workspace).await?;
    let data_dir = if let Some(data_dir) = data_dir {
        Some(fs::load(cache, data_dir).await?)
    } else {
        None
    };

    Ok((workspace, data_dir))
}
