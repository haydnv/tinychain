use std::path::PathBuf;
use std::sync::Arc;

use crate::context::TCResult;
use crate::error;
use crate::value::Link;

pub struct Drive {
    mount_point: PathBuf,
}

impl Drive {
    pub fn new(mount_point: PathBuf) -> Arc<Drive> {
        Arc::new(Drive { mount_point })
    }

    pub fn fs_path(self: Arc<Self>, context: Link, name: Link) -> TCResult<PathBuf> {
        if name.len() != 1 {
            return Err(error::bad_request("Block must be a Link of length 1", name));
        }

        let mut path = self.mount_point.clone();
        for dir in context.into_iter() {
            path.push(&dir.as_str()[1..]);
        }
        path.push(&name[0]);
        Ok(path)
    }
}
