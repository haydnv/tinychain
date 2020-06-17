use std::collections::hash_map::HashMap;

use crate::internal::cache;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::TxnId;
use crate::value::link::PathSegment;

pub type BlockId = PathSegment;

pub struct Block<'a> {
    file: &'a File<'a>,
    cached: cache::Block
}

impl<'a> Mutate for Block<'a> {
    fn diverge(&self, txn_id: &TxnId) -> Self {
        self.file.version(self.cached.name().clone(), txn_id)
    }

    fn converge(&mut self, _other: Block<'a>) {
        // TODO
    }
}

#[derive(Clone)]
struct FileContents<'a>(HashMap<BlockId, TxnLock<Block<'a>>>);

impl<'a> Mutate for FileContents<'a> {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    fn converge(&mut self, _other: FileContents) {
        // TODO
    }
}

pub struct File<'a> {
    cache: cache::Dir,
    contents: TxnLock<FileContents<'a>>,
}

impl<'a> File<'a> {
    fn version(&'a self, _name: BlockId, _txn_id: &TxnId) -> Block<'a> {
        // TODO
        panic!("NOT IMPLEMENTED")
    }
}
