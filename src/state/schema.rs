use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::cache;
use crate::internal::file::*;
use crate::internal::{Chain, FsDir};
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue, Version};

#[derive(Clone, Deserialize, Serialize)]
pub struct Schema {
    pub key: Vec<(String, Link)>,
    pub columns: Vec<(String, Link)>,
    pub version: Version,
}

impl Schema {
    pub fn as_map(&self) -> HashMap<String, Link> {
        let mut map: HashMap<String, Link> = HashMap::new();
        for (name, ctr) in &self.key {
            map.insert(name.clone(), ctr.clone());
        }
        for (name, ctr) in &self.columns {
            map.insert(name.clone(), ctr.clone());
        }

        map
    }
}

impl TryFrom<TCValue> for Schema {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Schema> {
        let value: Vec<TCValue> = value.try_into()?;
        if value.len() == 3 {
            let key: Vec<(String, Link)> = value[0].clone().try_into()?;
            let columns: Vec<(String, Link)> = value[1].clone().try_into()?;
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
    txn_cache: cache::Map<TransactionId, Schema>,
}

impl SchemaHistory {
    pub fn new(txn: &Arc<Transaction>, schema: Schema) -> TCResult<Arc<SchemaHistory>> {
        let txn_cache = cache::Map::new();
        txn_cache.insert(txn.id(), schema);

        Ok(Arc::new(SchemaHistory {
            chain: Chain::new(txn.context().reserve(&Link::to("/schema")?)?),
            txn_cache,
        }))
    }

    async fn commit(&self, txn_id: TransactionId) {
        if let Some(schema) = self.txn_cache.get(&txn_id) {
            self.chain.clone().put(&txn_id, &[schema]).await;
        }
    }

    pub async fn current(&self, txn_id: TransactionId) -> TCResult<Schema> {
        if let Some(schema) = self.txn_cache.get(&txn_id) {
            Ok(schema)
        } else if let Some(schema) = self
            .chain
            .until(txn_id.clone())
            .fold(None, |_: Option<Schema>, s: Vec<Schema>| {
                future::ready(if let Some(s) = s.last() {
                    Some(s.clone())
                } else {
                    None
                })
            })
            .await
        {
            Ok(schema)
        } else {
            Err(error::bad_request(
                "This table did not exist at transaction",
                txn_id,
            ))
        }
    }
}

#[async_trait]
impl File for SchemaHistory {
    async fn copy_from(reader: &mut FileReader, dest: Arc<FsDir>) -> TCResult<Arc<SchemaHistory>> {
        let (path, blocks) = reader.next().await.unwrap();
        let chain: Arc<Chain> = Chain::from(blocks, dest.reserve(&path)?).await;

        Ok(Arc::new(SchemaHistory {
            chain,
            txn_cache: cache::Map::new(),
        }))
    }

    async fn copy_to(&self, txn_id: TransactionId, writer: &mut FileWriter) -> TCResult<()> {
        writer.write_file(
            Link::to("/schema")?,
            Box::new(self.chain.into_stream(txn_id).boxed()),
        );
        Ok(())
    }
}
