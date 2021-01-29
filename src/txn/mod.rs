use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use auth::Token;
use error::*;
use generic::Id;
use transact::fs::{self, File};
pub use transact::{Transact, Transaction, TxnId};

use crate::state::chain::ChainBlock;

mod server;

pub use server::*;

pub struct Request {
    pub auth: Token,
    pub txn_id: TxnId,
}

impl Request {
    pub fn new(auth: Token, txn_id: TxnId) -> Self {
        Self { auth, txn_id }
    }

    pub fn contains(&self, other: &Self) -> bool {
        if self.txn_id == other.txn_id {
            self.auth.contains(&other.auth)
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub enum FileEntry {
    Chain(File<ChainBlock>),
}

impl fs::FileEntry for FileEntry {}

#[async_trait]
impl Transact for FileEntry {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Chain(chain) => chain.commit(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Chain(chain) => chain.finalize(txn_id).await,
        }
    }
}

impl From<File<ChainBlock>> for FileEntry {
    fn from(file: File<ChainBlock>) -> Self {
        Self::Chain(file)
    }
}

impl TryFrom<FileEntry> for File<ChainBlock> {
    type Error = TCError;

    fn try_from(entry: FileEntry) -> Result<Self, Self::Error> {
        match entry {
            FileEntry::Chain(chain) => Ok(chain),
        }
    }
}

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(_) => f.write_str("(a Chain file)"),
        }
    }
}

struct Inner {
    request: Request,
}

#[derive(Clone)]
pub struct Txn {
    inner: Arc<Inner>,
}

impl Txn {
    fn new(request: Request) -> Self {
        let inner = Arc::new(Inner { request });
        Self { inner }
    }

    pub fn request(&'_ self) -> &'_ Request {
        &self.inner.request
    }
}

#[async_trait]
impl Transaction<FileEntry> for Txn {
    fn id(&self) -> &TxnId {
        &self.inner.request.txn_id
    }

    async fn context<B: fs::BlockData>(&self) -> TCResult<File<B>> {
        unimplemented!()
    }

    async fn subcontext(&self, _id: Id) -> TCResult<Self> {
        unimplemented!()
    }
}
