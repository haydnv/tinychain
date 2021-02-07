use std::path::PathBuf;

use error::*;

mod block;
mod cache;
mod dir;
mod file;

pub use block::*;
pub use dir::*;
pub use file::*;

pub async fn mount(mount_point: PathBuf, cache_size: usize) -> TCResult<Dir> {
    let cache = cache::Cache::load(mount_point, cache_size).await?;
    Ok(Dir::load(cache.root()).await)
}
