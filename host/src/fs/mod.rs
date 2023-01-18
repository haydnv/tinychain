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

// TODO: move to the error crate & impl From<io::Error> for TCError
pub(crate) fn io_err(err: io::Error) -> TCError {
    match err.kind() {
        io::ErrorKind::WouldBlock => conflict!("synchronous filesystem access failed").consume(err),
        io::ErrorKind::NotFound => TCError::not_found(err),
        io::ErrorKind::PermissionDenied => {
            unexpected!("host filesystem permission denied").consume(err)
        }
        kind => unexpected!("host filesystem error: {:?}", kind).consume(err),
    }
}
