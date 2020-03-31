use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use crate::context::{Link, TCContext, TCResult, TCState, TCValue};
use crate::error;
use crate::host::Host;
use crate::state::chain::ChainContext;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Table {
    schema: Vec<(TCValue, TCValue)>,
    key: String,
}

impl Table {
    async fn insert(self: Arc<Self>, _key: TCValue, _value: TCValue) -> TCResult<()> {
        Err(error::not_implemented())
    }

    async fn update(self: Arc<Self>, _key: TCValue, _value: TCValue) -> TCResult<()> {
        Err(error::not_implemented())
    }

    async fn delete(self: Arc<Self>, _key: TCValue, _value: TCValue) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCContext for Table {
    async fn post(
        self: Arc<Self>,
        _host: Arc<Host>,
        method: String,
        txn: Arc<Transaction>,
    ) -> TCResult<Arc<TCState>> {
        match method.as_str() {
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

    async fn new_table(host: Arc<Host>, schema: Vec<TCValue>, key: String) -> TCResult<Table> {
        let mut valid_columns: Vec<(TCValue, TCValue)> = vec![];
        let mut key_present = false;

        for column in schema {
            let column = &TCValue::vector(&column)?;
            if let [TCValue::r#String(name), TCValue::Link(_)] = &column[..] {
                valid_columns.push((column[0].clone(), column[1].clone()));

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

        let mut data_types: Vec<Link> = vec![];
        for (_, link) in &valid_columns {
            data_types.push(TCValue::link(link)?);
        }
        let data_types = data_types.iter().map(|d| host.clone().get(d.clone()));
        for result in join_all(data_types).await {
            match result {
                Ok(_) => (),
                Err(cause) => {
                    return Err(cause);
                }
            }
        }

        Ok(Table {
            schema: valid_columns,
            key,
        })
    }
}

#[async_trait]
impl TCContext for TableContext {
    async fn post(
        self: Arc<Self>,
        host: Arc<Host>,
        method: String,
        txn: Arc<Transaction>,
    ) -> TCResult<Arc<TCState>> {
        if method != "new" {
            return Err(error::bad_request(
                "TableContext has no such method",
                method,
            ));
        }

        let key = TCValue::string(&TCState::value(txn.clone().require("key")?)?)?;
        let schema = TCValue::vector(&TCState::value(txn.require("schema")?)?)?;

        Ok(Arc::new(TCState::Table(Arc::new(
            Self::new_table(host, schema, key).await?,
        ))))
    }
}
