use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::lock::Mutex;
use futures::{future, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{Chain, Mutation};
use crate::internal::file::*;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::{TCResult, TCValue, ValueId, Version};

#[derive(Clone, Deserialize, Serialize)]
pub struct Schema {
    pub key: Vec<(ValueId, TCPath)>,
    pub columns: Vec<(ValueId, TCPath)>,
    pub version: Version,
}

impl Schema {
    pub fn as_map(&self) -> HashMap<ValueId, TCPath> {
        let mut map: HashMap<ValueId, TCPath> = HashMap::new();
        for (name, ctr) in &self.key {
            map.insert(name.clone(), ctr.clone());
        }
        for (name, ctr) in &self.columns {
            map.insert(name.clone(), ctr.clone());
        }

        map
    }

    pub fn from(
        key: Vec<(ValueId, TCPath)>,
        columns: Vec<(ValueId, TCPath)>,
        version: Version,
    ) -> Schema {
        Schema {
            key,
            columns,
            version,
        }
    }

    fn new() -> Schema {
        Schema {
            key: vec![],
            columns: vec![],
            version: "0.0.0".parse().unwrap(),
        }
    }
}

impl Mutation for Schema {}

impl TryFrom<TCValue> for Schema {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Schema> {
        let value: Vec<TCValue> = value.try_into()?;
        if value.len() == 3 {
            let key: Vec<(ValueId, TCPath)> = value[0].clone().try_into()?;
            let columns: Vec<(ValueId, TCPath)> = value[1].clone().try_into()?;
            let version: Version = value[2].clone().try_into()?;
            Ok(Schema {
                key,
                columns,
                version,
            })
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Schema of the form ([(name, constructor)...], [(name, constructor), ...], Version), found", value))
        }
    }
}

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
            .fold(Schema::new(), |_, s| future::ready(s))
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
