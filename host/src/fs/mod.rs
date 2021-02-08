use std::path::PathBuf;

use futures::TryFutureExt;
use futures_locks::RwLock;

use error::*;

mod cache;
mod dir;
mod file;

pub use cache::*;
pub use dir::*;
pub use file::*;

pub async fn load(cache: Cache, mount_point: PathBuf) -> TCResult<RwLock<CacheDir>> {
    let dir = CacheDir::load(cache.clone(), mount_point.clone())
        .map_ok(RwLock::new)
        .await?;

    cache.register(mount_point, dir.clone()).await;
    Ok(dir)
}
