use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::Map;
use crate::internal::chain::{Chain, ChainBlock, Mutation};
use crate::internal::file::*;
use crate::state::Transact;
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

    fn new() -> Schema {
        Schema {
            key: vec![],
            columns: vec![],
            version: Version::parse("0.0.0").unwrap(),
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
        println!("getting schema at {}", txn_id);
        if let Some(schema) = self.txn_cache.get(txn_id) {
            return schema;
        }

        let mut schema = Schema::new();
        println!("streaming past mutations from Chain");
        let mut stream = self.chain.stream_into::<Schema>(Some(txn_id.clone()));
        while let Some(s) = stream.next().await {
            println!("got past mutation");
            schema = s;
        }

        schema
    }

    pub async fn latest(&self) -> Schema {
        let mut schema = Schema::new();
        let mut stream = self.chain.stream_into::<Schema>(None);
        while let Some(s) = stream.next().await {
            schema = s;
        }

        schema
    }
}

#[async_trait]
impl File for SchemaHistory {
    type Block = ChainBlock<Schema>;

    async fn copy_into(&self, txn_id: TransactionId, copier: &mut FileCopier) {
        copier.write_file(
            "schema".try_into().unwrap(),
            Box::new(self.chain.stream_bytes::<Schema>(Some(txn_id)).boxed()),
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
impl Transact for SchemaHistory {
    async fn commit(&self, txn_id: &TransactionId) {
        println!("Schema::commit");
        if let Some(schema) = self.txn_cache.remove(txn_id) {
            self.chain.clone().put(txn_id, &[schema]).await;
        }
        println!("Schema::commit complete");
    }
}
