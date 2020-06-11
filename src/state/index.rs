use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::File;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{TCResult, TCType, Value};

use super::{Collect, GetResult};

pub struct Index {
    file: Arc<File>,
    schema: Vec<TCType>,
}

impl Index {
    async fn new(txn_id: &TxnId, schema: Vec<TCType>, file: Arc<File>) -> TCResult<Index> {
        if file.is_empty(txn_id).await {
            Ok(Index { file, schema })
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
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            ));
        }

        for (i, val) in key.iter().enumerate() {
            if !val.is_a(&self.schema[i]) {
                return Err(error::bad_request(
                    &format!("Expected {}", self.schema[i]),
                    &key[i],
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
