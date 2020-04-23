use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use futures::Future;
use tokio::fs;

use crate::error;
use crate::internal::cache::Map;
use crate::internal::DELIMITER;
use crate::value::{Link, TCResult};

#[derive(Debug)]
pub struct Dir {
    mount_point: PathBuf,
    context: Link,
    parent: Option<Arc<Dir>>,
    children: Map<Link, Arc<Dir>>,
    buffer: RwLock<HashMap<Link, Vec<u8>>>,
    tmp: bool,
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
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: false,
        })
    }

    pub fn new_tmp(context: Link, mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            mount_point,
            context,
            parent: None,
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: true,
        })
    }

    fn child(self: &Arc<Self>, context: Link, mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            mount_point,
            context,
            parent: Some(self.clone()),
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: self.tmp,
        })
    }

    pub fn reserve(self: &Arc<Self>, path: &Link) -> TCResult<Arc<Dir>> {
        if path == "/" {
            return Err(error::internal("Tried to reserve empty dir name"));
        }

        if path.len() == 1 {
            if self.children.contains_key(&path) {
                return Err(error::internal(&format!(
                    "Tried to reserve a directory that's already reserved! {}",
                    path
                )));
            }

            let dir = self.child(path.clone(), self.fs_path(path));
            self.children.insert(path.nth(0), dir.clone());
            Ok(dir)
        } else {
            let dir = if let Some(dir) = self.children.get(&path.nth(0)) {
                dir
            } else {
                let child_path = path.nth(0);
                let dir = Dir::new(child_path.clone(), self.fs_path(&child_path));
                self.children.insert(child_path, dir.clone());
                dir
            };

            dir.reserve(&path.slice_from(1))
        }
    }

    pub fn get(self: Arc<Self>, path: Link) -> impl Future<Output = TCResult<Vec<Vec<u8>>>> {
        println!("get file {}", path);
        async move {
            if let Some(buffer) = self.buffer.read().unwrap().get(&path) {
                let mut records: Vec<Vec<u8>> = buffer
                    .split(|b| *b == DELIMITER as u8)
                    .map(|c| c.to_vec())
                    .collect();
                records.pop();
                Ok(records)
            } else {
                // TODO
                Ok(vec![])
            }
        }
    }

    pub async fn append(self: &Arc<Self>, path: Link, data: Vec<u8>) -> TCResult<()> {
        println!("append to file {}", path);
        if data.contains(&(DELIMITER as u8)) {
            let msg = "Attempted to write a block containing the ASCII record delimiter (0x30)";
            return Err(error::internal(msg));
        }

        let data = [&data[..], &[DELIMITER as u8]].concat();

        let mut buffer = self.buffer.write().unwrap();
        match buffer.get_mut(&path) {
            Some(file_buffer) => file_buffer.extend(data),
            None => {
                buffer.insert(path, data);
            }
        }

        Ok(())
    }

    pub async fn exists(self: &Arc<Self>, path: &Link) -> TCResult<bool> {
        println!("check exists {}", path);
        let fs_path = self.fs_path(path);
        if self.children.contains_key(path) {
            println!("found it");
            return Ok(true);
        }

        match fs::metadata(fs_path).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    pub async fn flush(self: &Arc<Self>, _path: Link, _header: Vec<u8>, _data: Vec<Vec<u8>>) {
        if self.tmp {
            return;
        }

        // TODO
    }

    fn fs_path(&self, name: &Link) -> PathBuf {
        if !name.len() == 1 {
            panic!("Tried to look up the filesystem path of {}", name);
        }

        let mut path = self.mount_point.clone();

        for dir in self.context.clone().into_iter() {
            path.push(&dir.to_string()[1..]);
        }

        for i in 0..name.len() {
            path.push(name.as_str(i));
        }

        path
    }
}
