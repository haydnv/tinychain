use std::path::PathBuf;

use crate::lock::RwLock;

mod dir;

pub type Dir = dir::Dir;

pub fn mount(mount_point: PathBuf) -> RwLock<Dir> {
    RwLock::new(Dir::new(mount_point))
}
