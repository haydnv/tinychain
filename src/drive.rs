use std::path::PathBuf;
use std::sync::Arc;

pub struct Drive {
    mount_point: PathBuf,
}

impl Drive {
    pub fn new(mount_point: PathBuf) -> Arc<Drive> {
        Arc::new(Drive { mount_point })
    }
}
