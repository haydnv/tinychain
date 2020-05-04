use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::Map;
use crate::internal::file::*;
use crate::internal::Chain;
use crate::state::Transactable;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{TCPath, TCResult, TCValue, ValueId, Version};

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
}

impl Schema {
    fn new() -> Schema {
        Schema {
            key: vec![],
            columns: vec![],
            version: Version::parse("0.0.0").unwrap(),
        }
    }
}

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
    chain: Arc<Chain>,
    txn_cache: Map<TransactionId, Schema>,
}

impl SchemaHistory {
    pub fn new(txn: &Arc<Transaction>, schema: Schema) -> TCResult<Arc<SchemaHistory>> {
        let txn_cache = Map::new();
        txn_cache.insert(txn.id(), schema);

        let schema_history = Arc::new(SchemaHistory {
            chain: Chain::new(txn.context().reserve("schema")?),
            txn_cache,
        });
        txn.mutate(schema_history.clone());
        Ok(schema_history)
    }

    pub async fn at(&self, txn_id: &TransactionId) -> Schema {
        if let Some(schema) = self.txn_cache.get(txn_id) {
            return schema;
        }

        self.chain
            .stream_into_until(txn_id.clone())
            .fold(None, |_: Option<Schema>, s: Vec<Schema>| {
                future::ready(s.last().cloned())
            })
            .await
            .unwrap_or_else(Schema::new)
    }

    pub async fn latest(&self) -> Schema {
        self.chain
            .stream_into()
            .fold(None, |_: Option<Schema>, s: Vec<Schema>| {
                future::ready(s.last().cloned())
            })
            .await
            .unwrap_or_else(Schema::new)
    }
}

#[async_trait]
impl File for SchemaHistory {
    async fn copy_into(&self, txn_id: TransactionId, copier: &mut FileCopier) {
        copier.write_file(
            "schema".try_into().unwrap(),
            Box::new(self.chain.stream_until(txn_id).boxed()),
        );
    }

    async fn copy_from(copier: &mut FileCopier, dest: Arc<Store>) -> Arc<SchemaHistory> {
        let (path, blocks) = copier.next().await.unwrap();
        let chain: Arc<Chain> = Chain::copy_from(blocks, dest.reserve(path).unwrap()).await;

        Arc::new(SchemaHistory {
            chain,
            txn_cache: Map::new(),
        })
    }

    async fn from_store(store: Arc<Store>) -> Arc<SchemaHistory> {
        Arc::new(SchemaHistory {
            chain: Chain::from_store(store).await.unwrap(),
            txn_cache: Map::new(),
        })
    }
}

#[async_trait]
impl Transactable for SchemaHistory {
    async fn commit(&self, txn_id: &TransactionId) {
        if let Some(schema) = self.txn_cache.remove(txn_id) {
            self.chain.clone().put(txn_id, &[schema]).await;
        }
    }
}
