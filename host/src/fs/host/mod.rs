use std::fs::Metadata;
use std::path::PathBuf;

use futures::TryFutureExt;
use log::debug;
use tokio::fs;

use tc_error::*;
use tcgeneric::PathSegment;

use super::io_err;

pub type DirContents = Vec<(fs::DirEntry, Metadata)>;

pub async fn create_parent(path: &PathBuf) -> TCResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            tokio::fs::create_dir_all(parent)
                .map_err(|e| io_err(e, parent))
                .await?;
        }
    }

    Ok(())
}

pub async fn dir_contents(dir_path: &PathBuf) -> TCResult<Vec<(fs::DirEntry, Metadata)>> {
    let mut contents = vec![];
    let mut handles = fs::read_dir(dir_path)
        .map_err(|e| io_err(e, dir_path))
        .await?;

    while let Some(handle) = handles
        .next_entry()
        .map_err(|e| io_err(e, dir_path))
        .await?
    {
        if handle
            .path()
            .file_name()
            .expect("file name")
            .to_str()
            .expect("file name")
            .starts_with('.')
        {
            debug!("skip hidden file {:?}", handle.path());
            continue;
        }

        let meta = handle
            .metadata()
            .map_err(|e| io_err(e, handle.path()))
            .await?;

        contents.push((handle, meta));
    }

    Ok(contents)
}

#[inline]
pub fn file_ext(path: &'_ PathBuf) -> Option<&'_ str> {
    path.extension().and_then(|ext| ext.to_str())
}

pub fn file_name(handle: &fs::DirEntry) -> TCResult<PathSegment> {
    if let Some(name) = handle.path().file_stem() {
        let name = name.to_str().ok_or_else(|| {
            TCError::internal(format!("invalid file name at {:?}", handle.path()))
        })?;

        name.parse()
    } else {
        Err(TCError::internal("Cannot load file with no name!"))
    }
}

#[inline]
pub fn fs_path(mount_point: &PathBuf, name: &PathSegment) -> PathBuf {
    let mut path = mount_point.clone();
    path.push(name.to_string());
    path
}
