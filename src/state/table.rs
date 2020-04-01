use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Table {
    chain: Arc<Chain>,
    schema: Vec<(String, Link)>,
    key: String,
}

#[derive(Deserialize, Serialize)]
enum Delta {
    Insert(TCValue, TCValue),
}

impl Table {
    async fn insert(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        key: TCValue,
        value: TCValue,
    ) -> TCResult<()> {
        let delta = Delta::Insert(key, value);
        let delta = serde_json::to_string_pretty(&delta)?;
        let delta = TCValue::from_string(&delta);
        self.chain.clone().put(txn, delta).await
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

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, value: TCValue) -> TCResult<()> {
        let values = TCValue::vector(&value)?;
        let mut row: HashMap<String, TCValue> = HashMap::new();
        for value in values {
            let value = TCValue::vector(&value)?;
            if let [TCValue::r#String(column), value] = &value[..] {
                row.insert(column.clone(), value.clone());
            }
        }

        if !row.contains_key(&self.key) {
            return Err(error::bad_request(
                "You must specify the key of the row",
                self.key.clone(),
            ));
        }

        // TODO

        Err(error::not_implemented())
    }

    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, method: Link) -> TCResult<Arc<TCState>> {
        match method.as_str() {
            "/update" => Err(error::not_implemented()),
            "/delete" => Err(error::not_implemented()),
            _ => Err(error::bad_request("Table has no such method", method)),
        }
    }
}

pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
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

        let new_chain = Link::to("/sbin/chain/new")?;
        Ok(Table {
            chain: TCState::chain(txn.post(new_chain).await?)?,
            schema: valid_columns,
            key,
        })
    }
}

#[async_trait]
impl TCContext for TableContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<Arc<TCState>> {
        if method.as_str() != "/new" {
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
