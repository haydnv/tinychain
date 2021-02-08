use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_locks::RwLock;

use error::{TCError, TCResult};
use generic::{Id, PathSegment};
use transact::TxnId;

use crate::chain::ChainBlock;

use super::cache::CacheDir;
use super::FileView;

#[derive(Clone)]
pub enum FileEntry {
    Chain(FileView<ChainBlock>),
}

#[derive(Clone)]
pub enum DirEntry {
    Dir(RwLock<DirView>),
    File(FileEntry),
}

impl TryFrom<DirEntry> for FileView<ChainBlock> {
    type Error = TCError;

    fn try_from(dir: DirEntry) -> TCResult<FileView<ChainBlock>> {
        match dir {
            DirEntry::File(file) => match file {
                FileEntry::Chain(chain) => Ok(chain),
            },
            other => Err(TCError::bad_request(
                "expected a Chain file but found",
                other,
            )),
        }
    }
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dir(_) => f.write_str("a transaction-scoped directory"),
            Self::File(file) => match file {
                FileEntry::Chain(_) => f.write_str("a transaction-scoped chain file"),
            },
        }
    }
}

pub struct DirView {
    txn_id: TxnId,
    version: RwLock<CacheDir>,
}

impl DirView {
    pub fn new(txn_id: TxnId, version: RwLock<CacheDir>) -> Self {
        Self { txn_id, version }
    }
}

#[async_trait]
impl transact::fs::Dir for DirView {
    type Entry = DirEntry;

    async fn create_dir(&mut self, _name: PathSegment) -> TCResult<RwLock<Self>> {
        unimplemented!()
    }

    async fn create_file<F: transact::fs::File>(&mut self, _name: Id) -> TCResult<RwLock<F>>
    where
        Self::Entry: TryInto<F>,
    {
        unimplemented!()
    }

    async fn get_dir(&self, _name: &PathSegment) -> TCResult<Option<RwLock<Self>>> {
        unimplemented!()
    }

    async fn get_file<F: transact::fs::File>(&self, _name: &Id) -> TCResult<Option<RwLock<F>>>
    where
        Self::Entry: TryInto<F>,
    {
        unimplemented!()
    }
}

struct Inner {
    cache: RwLock<CacheDir>,
    versions: HashMap<TxnId, RwLock<DirView>>,
}

#[derive(Clone)]
pub struct Dir {
    inner: Arc<Inner>,
}

impl Dir {
    pub fn new(cache: RwLock<CacheDir>) -> Self {
        let inner = Inner {
            cache,
            versions: HashMap::new(),
        };

        Self {
            inner: Arc::new(inner),
        }
    }
}
