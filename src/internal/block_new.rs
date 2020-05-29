use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use futures::future::BoxFuture;

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
    children: HashMap<PathSegment, DirEntry>,
}

impl Dir {
    pub fn new(mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            context: mount_point,
            children: HashMap::new(),
        })
    }

    pub fn get_dir<'a>(&'a self, path: TCPath) -> BoxFuture<'a, Option<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                None
            } else if let Some(DirEntry::Dir(dir)) = self.children.get(&path[0]) {
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

    pub fn new_dir<'a>(&'a mut self, path: TCPath) -> BoxFuture<'a, TCResult<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::not_found(path))
            } else if path.len() == 1 {
                let path = path[0].clone();
                if self.children.contains_key(&path) {
                    Err(error::bad_request("Tried to create a new directory but there is already an entry at this path", &path))
                } else {
                    let dir = Dir::new(self.fs_path(&path));
                    self.children.insert(path, DirEntry::Dir(dir.clone()));
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
