use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use error::*;
use generic::{Id, PathSegment};
use transact::fs;
use transact::lock::{Mutable, TxnLock};

use crate::chain::ChainBlock;

use super::{dir_contents, file_name, fs_path, Cache, DirContents, File};

#[derive(Clone)]
pub enum FileEntry {
    Chain(File<ChainBlock>),
}

#[derive(Clone)]
pub enum DirEntry {
    Dir(Dir),
    File(FileEntry),
}

#[derive(Clone)]
pub struct Dir {
    cache: Cache,
    entries: TxnLock<Mutable<HashMap<PathSegment, DirEntry>>>,
}

impl Dir {
    fn load(
        cache: Cache,
        path: PathBuf,
        contents: DirContents,
    ) -> Pin<Box<dyn Future<Output = TCResult<Self>>>> {
        Box::pin(async move {
            if contents.iter().all(|(_, meta)| meta.is_dir()) {
                let mut entries = HashMap::new();

                for (handle, _) in contents.into_iter() {
                    let name = file_name(&handle)?;
                    let path = fs_path(&path, &name);
                    let contents = dir_contents(&path).await?;
                    if contents.iter().all(|(_, meta)| meta.is_file()) {
                        // TODO: support other file types
                        let file = File::load(cache.clone(), path, contents).await?;
                        entries.insert(name, DirEntry::File(FileEntry::Chain(file)));
                    } else if contents.iter().all(|(_, meta)| meta.is_dir()) {
                        let dir = Dir::load(cache.clone(), path, contents).await?;
                        entries.insert(name, DirEntry::Dir(dir));
                    } else {
                        return Err(TCError::internal(format!(
                            "directory at {:?} contains both blocks and subdirectories",
                            path
                        )));
                    }
                }

                Ok(Dir {
                    cache,
                    entries: TxnLock::new(format!("directory at {:?}", &path), entries.into()),
                })
            } else {
                Err(TCError::internal(format!(
                    "directory at {:?} contains both blocks and subdirectories",
                    path
                )))
            }
        })
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type File = FileEntry;

    async fn create_dir(&self, _name: PathSegment) -> TCResult<Self> {
        unimplemented!()
    }

    async fn create_file(&self, _name: Id) -> TCResult<Self::File> {
        unimplemented!()
    }

    async fn get_dir(&self, _name: &PathSegment) -> TCResult<Option<Self>> {
        unimplemented!()
    }

    async fn get_file(&self, _name: &Id) -> TCResult<Option<Self::File>> {
        unimplemented!()
    }
}

pub async fn load(cache: Cache, mount_point: PathBuf) -> TCResult<Dir> {
    let contents = dir_contents(&mount_point).await?;
    Dir::load(cache, mount_point, contents).await
}
