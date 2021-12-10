//! The transactional filesystem interface.

use std::io;
use std::path::Path;

use tc_error::*;
use tcgeneric::{label, Label};

pub use block::*;
pub use dir::*;
pub use file::*;

mod block;
mod dir;
mod file;

const VERSION: Label = label(".version");

#[inline]
fn file_ext(path: &'_ Path) -> Option<&'_ str> {
    path.extension().and_then(|ext| ext.to_str())
}

pub(crate) fn io_err(err: io::Error) -> TCError {
    match err.kind() {
        io::ErrorKind::NotFound => TCError::not_found(err),
        io::ErrorKind::PermissionDenied => TCError::internal(format!(
            "TinyChain does not have permission to access the host filesystem: {}",
            err
        )),
        kind => TCError::internal(format!("host filesystem error: {:?}: {}", kind, err)),
    }
}
