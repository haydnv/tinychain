use std::path::PathBuf;

mod dir;
mod lock;

pub type Dir = dir::Dir;
pub type RwLock<T> = lock::RwLock<T>;

pub fn mount(mount_point: PathBuf) -> RwLock<Dir> {
    RwLock::new(Dir::new(mount_point))
}
