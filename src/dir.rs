use std::sync::{Arc, RwLock};

use crate::context::*;
use crate::error;
use crate::transaction::Transaction;

#[derive(Clone)]
pub enum DirEntry {
    Dir(Arc<Dir>),
    Context(Arc<dyn TCContext>),
    Executable(Arc<dyn TCExecutable>),
    Object(Arc<dyn TCObject>),
}

pub struct Dir {
    entries: RwLock<Vec<(Link, DirEntry)>>,
}

impl Dir {
    pub fn new() -> Arc<Dir> {
        Arc::new(Dir {
            entries: RwLock::new(vec![]),
        })
    }

    pub fn get(self: Arc<Self>, _txn: Arc<Transaction>, path: &Link) -> TCResult<DirEntry> {
        let entries = self.entries.read().unwrap();
        for (entry_path, entry) in entries.iter() {
            if path == entry_path {
                return Ok(entry.clone());
            }
        }

        Err(error::not_found(path))
    }

    pub fn put_dir(self: Arc<Self>, rel_path: Link, dir: Arc<Dir>) {
        self.entries
            .write()
            .unwrap()
            .push((rel_path, DirEntry::Dir(dir)));
    }

    pub fn put_exe(self: Arc<Self>, rel_path: Link, exe: Arc<dyn TCExecutable>) {
        self.entries
            .write()
            .unwrap()
            .push((rel_path, DirEntry::Executable(exe)));
    }
}

impl Clone for Dir {
    fn clone(&self) -> Dir {
        Dir {
            entries: RwLock::new(self.entries.read().unwrap().clone()),
        }
    }
}
