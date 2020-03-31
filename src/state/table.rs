use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use crate::context::*;
use crate::error;
use crate::state::chain::{Chain, ChainContext};
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Table {
    chain: Arc<Chain>,
    schema: Vec<(String, Link)>,
    key: String,
}

impl Table {
    async fn insert(self: Arc<Self>, _key: TCValue, _value: TCValue) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCContext for Table {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, key: Link) -> TCResult<Arc<TCState>> {
        let _key = match &key.segments()[..] {
            [key] => key,
            _ => {
                return Err(error::bad_request("Invalid key specified for table", key));
            }
        };

        Err(error::not_implemented())
    }

    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: &str) -> TCResult<Arc<TCState>> {
        match method {
            "insert" => {
                let key = TCState::value(txn.clone().require("key")?)?;
                let value = TCState::value(txn.require("value")?)?;
                self.insert(key, value).await?;
                Ok(TCState::none())
            }
            "update" => Err(error::not_implemented()),
            "delete" => Err(error::not_implemented()),
            _ => Err(error::bad_request("Table has no such method", method)),
        }
    }
}

pub struct TableContext {
    chain_context: Arc<ChainContext>,
}

impl TableContext {
    pub fn new(chain_context: Arc<ChainContext>) -> Arc<TableContext> {
        Arc::new(TableContext { chain_context })
    }

    async fn new_table(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        schema: Vec<TCValue>,
        key: String,
    ) -> TCResult<Table> {
        let mut valid_columns: Vec<(String, Link)> = vec![];
        let mut key_present = false;

        for column in schema {
            let column = &TCValue::vector(&column)?;
            if let [TCValue::r#String(name), TCValue::Link(datatype)] = &column[..] {
                valid_columns.push((name.clone(), datatype.clone()));

                if name == &key {
                    key_present = true;
                }
            } else {
                return Err(error::bad_request(
                    "Expected a column definition but found",
                    format!("{:?}", column),
                ));
            }
        }

        if !key_present {
            return Err(error::bad_request("No such column was specified", key));
        }

        let data_types = valid_columns
            .iter()
            .map(|(_, d)| txn.clone().get(d.clone()));
        for result in join_all(data_types).await {
            match result {
                Ok(_) => (),
                Err(cause) => {
                    return Err(cause);
                }
            }
        }

        Ok(Table {
            chain: TCState::chain(self.chain_context.clone().post(txn, "new").await?)?,
            schema: valid_columns,
            key,
        })
    }
}

#[async_trait]
impl TCContext for TableContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: &str) -> TCResult<Arc<TCState>> {
        if method != "new" {
            return Err(error::bad_request(
                "TableContext has no such method",
                method,
            ));
        }

        let key = TCValue::string(&TCState::value(txn.clone().require("key")?)?)?;
        let schema = TCValue::vector(&TCState::value(txn.clone().require("schema")?)?)?;

        Ok(Arc::new(TCState::Table(Arc::new(
            self.new_table(txn, schema, key).await?,
        ))))
    }
}
