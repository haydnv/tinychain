use std::path::PathBuf;
use std::sync::Arc;

use crate::context::Link;

pub struct Drive {
    mount_point: PathBuf,
}

impl Drive {
    pub fn new(mount_point: PathBuf) -> Arc<Drive> {
        Arc::new(Drive { mount_point })
    }

    pub fn fs_path(self: Arc<Self>, context: Link, name: &str) -> PathBuf {
        let mut path = self.mount_point.clone();
        for dir in context.into_iter() {
            path.push(&dir.as_str()[1..]);
        }
        path.push(name);
        path
    }
}
