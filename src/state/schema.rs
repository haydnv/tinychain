use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::lock::Mutex;
use futures::{future, StreamExt};

use crate::internal::block::Store;
use crate::internal::chain::Chain;
use crate::internal::file::*;
use crate::state::table::Schema;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::TCResult;

pub struct SchemaHistory {
    chain: Mutex<Chain<Schema>>,
}

impl SchemaHistory {
    pub async fn new(txn: &Arc<Txn<'_>>, schema: Schema) -> TCResult<Arc<SchemaHistory>> {
        let chain = Chain::new(
            &txn.id(),
            txn.context().reserve(&txn.id(), "schema".parse()?).await?,
        )
        .await;
        chain.put(txn.id(), iter::once(schema)).await?;
        let schema_history = Arc::new(SchemaHistory {
            chain: Mutex::new(chain),
        });
        txn.mutate(schema_history.clone());
        Ok(schema_history)
    }

    pub async fn at(&self, txn_id: TxnId) -> Schema {
        self.chain
            .lock()
            .await
            .stream_into(txn_id)
            .fold(Schema::default(), |_, s| future::ready(s))
            .await
    }
}

#[async_trait]
impl File for SchemaHistory {
    async fn copy_into(&self, txn_id: TxnId, copier: &mut FileCopier) {
        copier.write_file(
            "schema".parse().unwrap(),
            Box::new(self.chain.lock().await.stream_bytes(txn_id).boxed()),
        );
    }

    async fn copy_from(
        copier: &mut FileCopier,
        txn_id: &TxnId,
        dest: Arc<Store>,
    ) -> Arc<SchemaHistory> {
        let (path, blocks) = copier.next().await.unwrap();
        let chain: Chain<Schema> =
            Chain::copy_from(blocks, txn_id, dest.reserve(txn_id, path).await.unwrap()).await;

        Arc::new(SchemaHistory {
            chain: Mutex::new(chain),
        })
    }

    async fn from_store(txn_id: &TxnId, store: Arc<Store>) -> Arc<SchemaHistory> {
        Arc::new(SchemaHistory {
            chain: Mutex::new(Chain::from_store(txn_id, store).await.unwrap()),
        })
    }
}

#[async_trait]
impl Transact for SchemaHistory {
    async fn commit(&self, txn_id: &TxnId) {
        self.chain.lock().await.commit(txn_id).await
    }
}
