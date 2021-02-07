use std::path::PathBuf;

use error::*;

mod block;
mod cache;
mod dir;
mod file;

pub use block::*;
pub use cache::Cache;
pub use dir::*;
pub use file::*;

pub async fn cache_init(mount_point: PathBuf, max_size: usize) -> TCResult<Cache> {
    cache::Cache::load(mount_point, max_size).await
}
