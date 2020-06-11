use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error;
use crate::internal::{BlockId, File};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{TCResult, TCType, Value, ValueId};

use super::{Collect, GetResult};

const BLOCK_SIZE: usize = 100_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit
const MAX_KEY_SIZE: usize = ((BLOCK_SIZE - BLOCK_ID_SIZE) / 2) - BLOCK_ID_SIZE;

#[derive(Deserialize, Serialize)]
struct Key {
    value: Vec<Value>,
    deleted: bool,
}

impl Key {
    fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    fn len(&self) -> usize {
        self.value.len()
    }
}

#[derive(Deserialize, Serialize)]
struct Node {
    leaf: bool,
    keys: Vec<Key>,
}

impl Node {
    fn new(leaf: bool) -> Node {
        Node { leaf, keys: vec![] }
    }
}

pub struct Column {
    name: ValueId,
    dtype: TCType,
    max_len: usize,
}

pub struct Index {
    file: Arc<File>,
    schema: Vec<Column>,
    order: usize,
    root: BlockId,
}

impl Index {
    async fn create(txn_id: TxnId, schema: Vec<Column>, file: Arc<File>) -> TCResult<Index> {
        // length-delimited serialization adds 32 bytes to each key as-stored
        let key_size: usize = 32 + schema.iter().map(|c| c.max_len).sum::<usize>();

        if key_size >= MAX_KEY_SIZE {
            return Err(error::bad_request(
                "The specified Index schema exceeds the maximum size",
                MAX_KEY_SIZE,
            ));
        }

        let order: usize = (BLOCK_SIZE - BLOCK_ID_SIZE) / (key_size + BLOCK_ID_SIZE);

        if file.is_empty(&txn_id).await {
            let root: BlockId = Uuid::new_v4().into();
            file.new_block(
                txn_id,
                root.clone(),
                Bytes::from(bincode::serialize(&Node::new(true))?),
            )
            .await?;

            Ok(Index {
                file,
                schema,
                order,
                root,
            })
        } else {
            Err(error::bad_request(
                "Tried to create a new Index without a new File",
                file,
            ))
        }
    }

    fn validate_key(&self, key: Vec<Value>) -> TCResult<()> {
        if self.schema.len() != key.len() {
            return Err(error::bad_request(
                &format!("Invalid key {}, expected", Value::Vector(key)),
                self.schema
                    .iter()
                    .map(|c| c.dtype.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            ));
        }

        for (i, column) in self.schema.iter().enumerate() {
            if !key[i].is_a(&column.dtype) {
                return Err(error::bad_request(
                    &format!("Expected {} for", column.dtype),
                    &column.name,
                ));
            }

            let key_size = bincode::serialized_size(&key[i])?;
            if key_size as usize > column.max_len {
                return Err(error::bad_request(
                    "Column value exceeds the maximum length",
                    &column.name,
                ));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Collect for Index {
    type Selector = Vec<Value>;
    type Item = Vec<Value>;

    async fn get(&self, _txn: &Arc<Txn>, _selector: &Self::Selector) -> GetResult {
        Err(error::not_implemented())
    }

    async fn put(
        &self,
        _txn: &Arc<Txn>,
        selector: Self::Selector,
        _value: Self::Item,
    ) -> TCResult<()> {
        self.validate_key(selector)?;

        Err(error::not_implemented())
    }
}

#[async_trait]
impl Transact for Index {
    async fn commit(&self, txn_id: &TxnId) {
        self.file.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.file.rollback(txn_id).await
    }
}

fn new_block_id(existing_ids: HashSet<ValueId>) -> ValueId {
    loop {
        let id: ValueId = Uuid::new_v4().into();
        if !existing_ids.contains(&id) {
            return id;
        }
    }
}
