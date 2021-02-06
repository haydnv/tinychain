use std::collections::hash_map::{Entry, HashMap};
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;

use bytes::Bytes;
use futures::future::{Future, TryFutureExt};
use futures::stream::{FuturesUnordered, TryStreamExt};
use futures_locks::RwLock;
use tokio::fs;

use error::*;
use generic::PathSegment;

pub enum DirEntry {
    Block(PathBuf),
    Dir(RwLock<Dir>),
}

pub struct Dir {
    mount_point: PathBuf,
    contents: HashMap<PathSegment, DirEntry>,
}

impl Dir {
    pub fn new(mount_point: PathBuf) -> Pin<Box<dyn Future<Output = TCResult<Dir>> + Send>> {
        Box::pin(async move {
            let mut contents = HashMap::new();

            let mut handles = fs::read_dir(&mount_point)
                .map_err(|e| io_err(e, &mount_point))
                .await?;

            let mut resolvers = FuturesUnordered::new();
            while let Some(handle) = handles
                .next_entry()
                .map_err(|e| io_err(e, &mount_point))
                .await?
            {
                let meta = handle
                    .metadata()
                    .map_err(|e| io_err(e, handle.path()))
                    .await?;
                let name = handle.file_name().to_str().unwrap().parse()?;
                let path = handle.path();

                resolvers.push(async move {
                    if meta.file_type().is_dir() {
                        let dir = Dir::new(path).await?;
                        Ok((name, DirEntry::Dir(RwLock::new(dir))))
                    } else if meta.file_type().is_file() {
                        Ok((name, DirEntry::Block(path)))
                    } else {
                        Err(TCError::not_implemented(
                            "Tinychain does not yet support symlinks",
                        ))
                    }
                });
            }

            while let Some((name, entry)) = resolvers.try_next().await? {
                contents.insert(name, entry);
            }

            Ok(Dir {
                mount_point,
                contents,
            })
        })
    }

    pub fn block_ids(&'_ self) -> impl Iterator<Item = &'_ PathSegment> + '_ {
        self.contents
            .iter()
            .filter_map(|(name, entry)| match entry {
                DirEntry::Block(_) => Some(name),
                _ => None,
            })
    }

    pub async fn create_block(&mut self, name: PathSegment, initial_value: Bytes) -> TCResult<()> {
        let block_path = fs_path(&self.mount_point, &name);
        fs::write(&block_path, initial_value)
            .map_err(|e| io_err(e, &block_path))
            .await?;

        self.contents.insert(name, DirEntry::Block(block_path));
        Ok(())
    }

    pub async fn copy_all(&mut self, source: &Dir) -> TCResult<()> {
        let block_ids = source.block_ids().cloned();
        for block_id in block_ids {
            self.copy_block(block_id, source).await?;
        }

        Ok(())
    }

    pub async fn copy_block(&mut self, name: PathSegment, source: &Dir) -> TCResult<()> {
        let block = source
            .get_block(&name)
            .await?
            .ok_or_else(|| TCError::not_found(&name))?;

        let path = fs_path(&self.mount_point, &name);
        fs::write(&path, &block)
            .map_err(|e| io_err(e, &path))
            .await?;

        self.contents.insert(name, DirEntry::Block(path));
        Ok(())
    }

    pub async fn create_dir(&mut self, name: PathSegment) -> TCResult<RwLock<Dir>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                let dir = Dir::new(fs_path(&self.mount_point, &name)).await?;
                let dir = RwLock::new(dir);
                entry.insert(DirEntry::Dir(dir.clone()));
                Ok(dir)
            }
        }
    }

    pub async fn create_or_get_dir(&mut self, name: &PathSegment) -> TCResult<RwLock<Dir>> {
        match self.get_dir(name) {
            Ok(Some(dir)) => Ok(dir),
            Err(cause) => Err(cause),
            Ok(None) => self.create_dir(name.clone()).await,
        }
    }

    pub fn delete(&mut self, name: &PathSegment) -> TCResult<()> {
        self.contents.remove(name);
        Ok(())
    }

    pub async fn delete_block(&mut self, name: &PathSegment) -> TCResult<()> {
        match self.contents.remove(name) {
            None => Err(TCError::not_found(name)),
            Some(DirEntry::Block(path)) => {
                fs::remove_file(&path).map_err(|e| io_err(e, &path)).await
            }
            Some(entry) => {
                self.contents.insert(name.clone(), entry);
                Err(TCError::bad_request(
                    "Expected filesystem block but found",
                    "(directory)",
                ))
            }
        }
    }

    pub async fn delete_dir(&mut self, name: &PathSegment) -> TCResult<()> {
        match self.contents.remove(name) {
            None => Ok(()),
            Some(DirEntry::Dir(dir)) => {
                let path = fs_path(&self.mount_point, name);
                fs::remove_file(&path).map_err(|e| io_err(e, &path)).await
            }
            Some(entry) => {
                self.contents.insert(name.clone(), entry);
                Err(TCError::bad_request(
                    "Expected filesystem directory but found",
                    "(block)",
                ))
            }
        }
    }

    pub async fn get_block(&self, name: &PathSegment) -> TCResult<Option<Bytes>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Dir(_)) => Err(TCError::bad_request("Not a block", name)),
            Some(DirEntry::Block(path)) => {
                fs::read(path)
                    .map_ok(Bytes::from)
                    .map_ok(Some)
                    .map_err(|e| io_err(e, path))
                    .await
            }
        }
    }

    pub fn get_dir(&self, name: &PathSegment) -> TCResult<Option<RwLock<Dir>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(DirEntry::Block(_)) => Err(TCError::bad_request(
                "Expected filesystem directory but found",
                "(block)",
            )),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

fn fs_path(mount_point: &PathBuf, name: &PathSegment) -> PathBuf {
    let mut path = mount_point.clone();
    path.push(name.to_string());
    path
}

fn io_err<I: fmt::Debug>(err: io::Error, info: I) -> TCError {
    match err.kind() {
        io::ErrorKind::NotFound => {
            TCError::unsupported(format!("There is already a directory at {:?}", info))
        }
        io::ErrorKind::PermissionDenied => TCError::internal(format!(
            "Tinychain does not have permission to access the host filesystem: {:?}",
            info
        )),
        other => TCError::internal(format!("host filesystem error: {:?}", other)),
    }
}
