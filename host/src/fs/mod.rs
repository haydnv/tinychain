//! The transactional filesystem interface.

use std::fmt;
use std::io;

use tc_error::*;
use tcgeneric::{label, Label};

mod block;
mod cache;
mod dir;
mod file;
mod host;

use host::{create_parent, dir_contents, file_ext, file_name, fs_path, DirContents};

pub use block::*;
pub use cache::*;
pub use dir::*;
pub use file::*;

const VERSION: Label = label(".version");

fn io_err<I: fmt::Debug + Send>(err: io::Error, info: I) -> TCError {
    match err.kind() {
        io::ErrorKind::NotFound => {
            TCError::internal(format!("host filesystem has no such entry {:?}", info))
        }
        io::ErrorKind::PermissionDenied => TCError::internal(format!(
            "Tinychain does not have permission to access the host filesystem: {:?}",
            info
        )),
        other => TCError::internal(format!("host filesystem error: {:?}: {}", other, err)),
    }
}
