use std::collections::HashMap;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

#[derive(Hash)]
pub struct Table {
    schema: Vec<(String, Link)>,
    key: (String, Link),
    chain: Arc<Chain>,
}

impl Table {
    fn schema_map(self: Arc<Self>) -> HashMap<String, Link> {
        HashMap::from_iter(self.schema.iter().cloned())
    }
}

#[derive(Deserialize, Serialize)]
pub enum TableRequest {
    Select,
    Insert,
    Update,
    Delete,
}

#[async_trait]
impl TCContext for Table {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _row_id: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _state: TCState) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCExecutable for Table {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, _method: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }
}

pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
    }

    async fn new_table<'a>(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        schema: Vec<(String, Link)>,
        key_column: String,
    ) -> TCResult<Arc<Table>> {
        let mut valid_columns: Vec<(String, Link)> = vec![];
        let mut key = None;

        for (name, datatype) in schema {
            valid_columns.push((name.clone(), datatype.clone()));

            if name == key_column {
                key = Some((name, datatype));
            }
        }

        let key = match key {
            Some(key) => key,
            None => {
                return Err(error::bad_request(
                    "No column was defined for the primary key",
                    key_column,
                ));
            }
        };

        let chain_path = txn.clone().context();
        txn.clone()
            .put(Link::to("/sbin/chain")?, TCState::Value(TCValue::Link(chain_path.clone())))
            .await?;
        let chain: Arc<Chain> = txn.get(chain_path.clone()).await?.to_chain()?;

        Ok(Arc::new(Table {
            key,
            schema: valid_columns,
            chain,
        }))
    }
}

#[async_trait]
impl TCExecutable for TableContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<TCState> {
        if method.as_str() != "/new" {
            return Err(error::bad_request(
                "TableContext has no such method",
                method,
            ));
        }

        let schema: Vec<(String, Link)> = txn.clone().require("schema")?;
        let key: String = txn.clone().require("key")?;
        Ok(TCState::Table(
            self.new_table(txn, schema, key).await?,
        ))
    }
}
