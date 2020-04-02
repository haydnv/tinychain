use std::collections::HashMap;
use std::iter::FromIterator;
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
    schema: Vec<(String, Link)>,
    key: (String, Link),
    chain: Arc<Chain>,
}

#[derive(Deserialize, Serialize)]
enum Delta {
    Insert(TCValue, Vec<(String, TCValue)>),
}

impl Delta {
    fn insert_from(key: String, row: &HashMap<String, TCValue>) -> TCResult<Delta> {
        if let Some(row_id) = row.get(&key) {
            let values = row
                .iter()
                .map(|(column, value)| (column.clone(), value.clone()))
                .collect();
            Ok(Delta::Insert(row_id.clone(), values))
        } else {
            Err(error::bad_request(
                "Cannot insert into a table without a value for the primary key",
                key,
            ))
        }
    }

    fn to_bytes(&self) -> TCResult<TCValue> {
        let serialized = serde_json::to_string_pretty(self)?;
        Ok(TCValue::from_bytes(serialized.as_bytes().to_vec()))
    }
}

impl Table {
    fn schema_map(self: Arc<Self>) -> HashMap<String, Link> {
        HashMap::from_iter(
            self.schema
                .iter()
                .map(|(name, constructor)| (name.clone(), constructor.clone())),
        )
    }
}

#[async_trait]
impl TCContext for Table {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, row_id: Link) -> TCResult<Arc<TCState>> {
        let _row_id = match &row_id.segments()[..] {
            [row_id] => row_id,
            _ => {
                return Err(error::bad_request(
                    "Invalid key specified for table",
                    row_id,
                ));
            }
        };

        // TODO

        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()> {
        let (key_col, _) = self.key.clone();
        let values = value.to_vec()?;
        let mut row: HashMap<String, TCValue> = HashMap::new();
        for value in values {
            let value = value.to_vec()?;
            if let [TCValue::r#String(column), value] = &value[..] {
                row.insert(column.clone(), value.clone());
            } else {
                return Err(error::bad_request(
                    "Expected [name, value], found",
                    format!("{:?}", value),
                ));
            }
        }

        if !row.contains_key(&key_col) {
            return Err(error::bad_request(
                "You must specify the key of the row",
                key_col,
            ));
        }

        if row.len() == 1 {
            return Err(error::bad_request(
                "You must specify at least one value to update",
                value,
            ));
        }

        let schema = self.clone().schema_map();
        let mut columns: Vec<String> = Vec::with_capacity(schema.len());
        let mut constructors: Vec<(Link, TCValue)> = Vec::with_capacity(schema.len());
        for (name, value) in row {
            if let Some(ctr) = schema.get(&name) {
                columns.push(name);
                constructors.push((ctr.clone(), value.clone()));
            } else {
                return Err(error::bad_request(
                    "Value specified for unknown column",
                    name,
                ));
            }
        }

        let results = join_all(
            constructors
                .iter()
                .map(|(c, v)| txn.clone().post(c.clone(), vec![("from", v.clone())])),
        )
        .await;

        let mut row: HashMap<String, TCValue> = HashMap::new();
        for i in 0..results.len() {
            match results[i].clone() {
                Ok(state) => {
                    let value = state.to_value()?;
                    row.insert(columns[i].clone(), value);
                }
                Err(cause) => {
                    return Err(cause);
                }
            }
        }

        let delta = Delta::insert_from(key_col, &row)?;
        self.chain.clone().put(txn, delta.to_bytes()?).await
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
        key_column: String,
    ) -> TCResult<Table> {
        let mut valid_columns: Vec<(String, Link)> = vec![];
        let mut key = None;

        for column in schema {
            let column = column.to_vec()?;
            if let [TCValue::r#String(name), TCValue::Link(datatype)] = &column[..] {
                valid_columns.push((name.clone(), datatype.clone()));

                if name == &key_column {
                    key = Some((name.clone(), datatype.clone()));
                }
            } else {
                return Err(error::bad_request(
                    "Expected a column definition but found",
                    format!("{:?}", column),
                ));
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

        let data_types = valid_columns
            .iter()
            .map(|(_, d)| txn.clone().extend(d.clone()).get());
        for result in join_all(data_types).await {
            match result {
                Ok(_) => (),
                Err(cause) => {
                    return Err(cause);
                }
            }
        }

        let chain = txn
            .post(Link::to("/sbin/chain/new")?, vec![])
            .await?
            .to_chain()?;

        Ok(Table {
            key,
            schema: valid_columns,
            chain,
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

        let key = txn.clone().require("key")?.to_value()?.to_string()?;
        let schema = txn.clone().require("schema")?.to_value()?.to_vec()?;

        Ok(Arc::new(TCState::Table(Arc::new(
            self.new_table(txn, schema, key).await?,
        ))))
    }
}
