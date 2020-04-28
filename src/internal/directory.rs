use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use crate::error;
use crate::internal::cache;
use crate::internal::file;
use crate::internal::{Chain, FsDir};
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue};

struct Directory {
    context: Arc<FsDir>,
    chain: Arc<Chain>,
    txn_cache: RwLock<HashMap<TransactionId, HashMap<Link, (Arc<FsDir>, State)>>>,
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
        txn: Arc<Transaction>,
        path: Self::Key,
        state: Self::Value,
    ) -> TCResult<Arc<Self>> {
        if path.len() != 1 {
            return Err(error::not_found(path));
        }

        let entry = (self.context.reserve(&path)?, state);
        if let Some(mutations) = self.txn_cache.write().unwrap().get_mut(&txn.id()) {
            mutations.insert(path, entry);
        } else {
            let mut mutations: HashMap<Link, (Arc<FsDir>, State)> = HashMap::new();
            mutations.insert(path, entry);
            self.txn_cache.write().unwrap().insert(txn.id(), mutations);
        }

        Ok(self)
    }
}

#[async_trait]
impl file::File for Directory {
    async fn copy_from(
        _reader: &mut file::FileReader,
        _dest: Arc<FsDir>,
    ) -> TCResult<Arc<Directory>> {
        Err(error::not_implemented())
    }

    async fn copy_to(
        &self,
        _txn_id: TransactionId,
        _writer: &mut file::FileWriter,
    ) -> TCResult<()> {
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
            txn_cache: RwLock::new(HashMap::new()),
        }))
    }
}
