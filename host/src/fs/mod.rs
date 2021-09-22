//! The transactional filesystem interface.

use std::fmt;
use std::io;
use std::path::Path;

use futures::TryFutureExt;
use log::debug;
use tokio::fs;

use tc_error::*;
use tcgeneric::{label, Label, PathSegment};

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

fn file_name(path: &Path) -> TCResult<PathSegment> {
    if let Some(name) = path.file_stem() {
        let name = name
            .to_str()
            .ok_or_else(|| TCError::internal(format!("invalid file name at {:?}", path)))?;

        name.parse()
    } else {
        Err(TCError::internal("Cannot load file with no name!"))
    }
}

pub fn io_err(err: io::Error) -> TCError {
    match err.kind() {
        io::ErrorKind::NotFound => TCError::not_found(err),
        io::ErrorKind::PermissionDenied => TCError::internal(format!(
            "TinyChain does not have permission to access the host filesystem: {}",
            err
        )),
        _ => TCError::internal(format!("host filesystem error: {}", err)),
    }
}
