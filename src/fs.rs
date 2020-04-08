use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::context::TCResult;
use crate::error;
use crate::value::Link;

pub struct Dir {
    mount_point: PathBuf,
    context: Link,
    parent: Option<Arc<Dir>>,
    children: RwLock<HashMap<Link, Arc<Dir>>>,
}

impl Drop for Dir {
    fn drop(&mut self) {
        if let Some(parent) = &self.parent {
            parent.children.write().unwrap().remove(&self.context);
        }
    }
}

impl Hash for Dir {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mount_point.hash(state);
    }
}

impl Dir {
    pub fn new(context: Link, mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            mount_point,
            context,
            parent: None,
            children: RwLock::new(HashMap::new())
        })
    }

    pub fn reserve(self: Arc<Self>, path: Link) -> TCResult<Arc<Dir>> {
        if self.children.read().unwrap().contains_key(&path) {
            return Err(error::internal(&format!("Tried to reserve a fs::Dir that's already reserved! {}", path)));
        }

        let dir = Dir::new(path.clone(), self.clone().fs_path(&path)?);
        self.children.write().unwrap().insert(path, dir.clone());
        Ok(dir)
    }

    fn fs_path(self: Arc<Self>, name: &Link) -> TCResult<PathBuf> {
        if name.len() != 1 {
            return Err(error::bad_request("Block path must be a Link of length 1", name));
        }

        let mut path = self.mount_point.clone();
        for dir in self.context.clone().into_iter() {
            path.push(&dir.as_str()[1..]);
        }
        path.push(&name[0]);
        Ok(path)
    }
}
