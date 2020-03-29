use std::path::PathBuf;

pub struct Drive {
    mount_point: PathBuf,
}

impl Drive {
    pub fn new(mount_point: PathBuf) -> Drive {
        Drive { mount_point }
    }
}
