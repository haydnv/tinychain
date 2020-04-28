use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::cache;
use crate::internal::file::*;
use crate::internal::{Chain, FsDir};
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue};

struct Directory {
    context: Arc<FsDir>,
    chain: Arc<Chain>,
    txn_cache: cache::Map<TransactionId, HashMap<Link, State>>,
}

#[async_trait]
impl Collection for Directory {
    type Key = Link;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        _key: &Self::Key,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _path: Self::Key,
        _state: Self::Value,
    ) -> TCResult<Arc<Self>> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl File for Directory {
    async fn copy_from(_reader: &mut FileReader, _dest: Arc<FsDir>) -> TCResult<Arc<Directory>> {
        Err(error::not_implemented())
    }

    async fn copy_to(&self, _txn_id: TransactionId, _writer: &mut FileWriter) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Persistent for Directory {
    type Config = TCValue; // TODO: permissions

    async fn commit(&self, _txn_id: TransactionId) {
        // TODO
    }

    async fn create(txn: Arc<Transaction>, _: TCValue) -> TCResult<Arc<Directory>> {
        Ok(Arc::new(Directory {
            context: txn.context(),
            chain: Chain::new(txn.context().reserve(&Link::to("/.contents")?)?),
            txn_cache: cache::Map::new(),
        }))
    }
}
