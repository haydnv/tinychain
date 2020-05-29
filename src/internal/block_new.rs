use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::lock::Mutex;

use crate::error;
use crate::value::link::{PathSegment, TCPath};
use crate::value::TCResult;

pub type BlockId = PathSegment;

enum DirEntry {
    Dir(Arc<Dir>),
    Store(Arc<Store>),
}

pub struct Dir {
    context: PathBuf,
    children: Mutex<HashMap<PathSegment, DirEntry>>,
}

impl Dir {
    pub fn new(mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            context: mount_point,
            children: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_dir<'a>(&'a self, path: TCPath) -> BoxFuture<'a, Option<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                None
            } else if let Some(DirEntry::Dir(dir)) = self.children.lock().await.get(&path[0]) {
                if path.len() == 1 {
                    Some(dir.clone())
                } else {
                    dir.get_dir(path.slice_from(1)).await
                }
            } else {
                None
            }
        })
    }

    pub fn get_or_create_dir<'a>(&'a self, path: TCPath) -> BoxFuture<'a, TCResult<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                return Err(error::bad_request("Not a valid directory name", path));
            }

            match self.children.lock().await.get(&path[0]) {
                Some(DirEntry::Store(_)) => Err(error::bad_request(
                    "Requested a Directory but found a Store",
                    path,
                )),
                Some(DirEntry::Dir(dir)) if path.len() == 1 => Ok(dir.clone()),
                Some(DirEntry::Dir(dir)) => dir.get_or_create_dir(path.slice_from(1)).await,
                None => self.new_dir(path).await,
            }
        })
    }

    pub fn new_dir<'a>(&'a self, path: TCPath) -> BoxFuture<'a, TCResult<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Not a valid directory name", path))
            } else if path.len() == 1 {
                let path = path[0].clone();
                let mut children = self.children.lock().await;
                if children.contains_key(&path) {
                    Err(error::bad_request("Tried to create a new directory but there is already an entry at this path", &path))
                } else {
                    let dir = Dir::new(self.fs_path(&path));
                    children.insert(path, DirEntry::Dir(dir.clone()));
                    Ok(dir)
                }
            } else {
                self.new_dir(path.slice_from(1)).await
            }
        })
    }

    fn fs_path(&self, name: &PathSegment) -> PathBuf {
        let mut path = self.context.clone();
        path.push(name.to_string());
        path
    }
}

pub struct Store {}
