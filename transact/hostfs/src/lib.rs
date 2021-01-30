use std::path::PathBuf;

use futures_locks::RwLock;

mod dir;

pub use dir::Dir;

pub async fn mount(mount_point: PathBuf) -> RwLock<Dir> {
    RwLock::new(Dir::new(mount_point))
}
