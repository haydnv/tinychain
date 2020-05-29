use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::value::link::{PathSegment, TCPath};

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
    pub fn new(mount_point: PathBuf) -> Dir {
        Dir {
            context: mount_point,
            children: HashMap::new(),
        }
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
}

pub struct Store {}
