use std::collections::HashMap;

use futures_locks::RwLock;

use transact::TxnId;

use crate::chain::ChainBlock;

use super::cache::CacheDir;
use super::File;

#[derive(Clone)]
enum FileEntry {
    Chain(File<ChainBlock>),
}

#[derive(Clone)]
enum DirEntry {
    Dir(RwLock<Dir>),
    File(FileEntry),
}

pub struct DirView {
    txn_id: TxnId,
    cache: RwLock<CacheDir>,
}

pub struct Dir {
    cache: RwLock<CacheDir>,
    versions: HashMap<TxnId, DirView>,
}

impl Dir {
    pub async fn load(_cache: RwLock<CacheDir>) -> Self {
        unimplemented!()
    }
}
