use std::path::PathBuf;

use futures::TryFutureExt;
use futures_locks::RwLock;

use error::*;

mod dir;

pub use dir::Dir;

pub async fn mount(mount_point: PathBuf) -> TCResult<RwLock<Dir>> {
    Dir::new(mount_point).map_ok(RwLock::new).await
}
