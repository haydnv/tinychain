use std::collections::HashMap;
use std::convert::TryInto;
use std::iter::{repeat, FromIterator};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

struct Mutation {
    schema: HashMap<String, usize>,
    key: TCValue,
    values: Vec<Option<TCValue>>,
}

impl Mutation {
    fn apply(&mut self, values: Vec<Option<TCValue>>) {
        for (i, value) in values.iter().enumerate() {
            if let Some(value) = value {
                self.values[i] = Some(value.clone());
            }
        }
    }

    fn set(&mut self, column: String, value: TCValue) -> TCResult<()> {
        if let Some(index) = self.schema.get(&column) {
            self.values[*index] = Some(value);
            Ok(())
        } else {
            Err(error::internal(
                "Table attempted to mutate nonexistent column",
            ))
        }
    }

    fn values(&self) -> TCValue {
        self.values.clone().into_iter().collect()
    }
}

#[derive(Debug, Hash)]
pub struct Table {
    key: (String, Link),
    columns: Vec<(String, Link)>,
    chain: Arc<Chain>,
}

impl Table {
    fn mutation(&self, key: TCValue) -> Mutation {
        let mut schema: Vec<(String, usize)> = vec![];
        for i in 0..self.columns.len() {
            schema.push((self.columns[i].0.to_owned(), i));
        }
        let schema: HashMap<String, usize> = schema.into_iter().collect();

        Mutation {
            schema,
            key,
            values: repeat(None).take(self.columns.len()).collect(),
        }
    }

    fn schema_map(&self) -> HashMap<String, Link> {
        HashMap::from_iter(self.columns.iter().cloned())
    }
}

#[async_trait]
impl TCContext for Table {
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, row_id: TCValue) -> TCResult<TCState> {
        let mut row = self.mutation(row_id.clone());
        let mutations: Vec<TCValue> = self.chain.clone().get(txn, row_id).await?.try_into()?;
        for mutation in mutations {
            let mutation: Vec<Option<TCValue>> = mutation.try_into()?;
            row.apply(mutation);
        }

        Ok(row.values().into())
    }

    async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        row_id: TCValue,
        column_values: TCState,
    ) -> TCResult<TCState> {
        let column_values: Vec<TCValue> = column_values.try_into()?;
        let schema = self.schema_map();

        let mut row = vec![];
        row.push(txn.clone().post(&self.key.1, vec![("from", row_id)]));
        for column_value in column_values.iter() {
            let column_value: (String, TCValue) = column_value.clone().try_into()?;
            let (column, value) = column_value;

            if let Some(ctr) = schema.get(&column) {
                row.push(txn.clone().post(ctr, vec![("from", value)]));
            } else {
                return Err(error::bad_request(
                    "This table contains no such column",
                    column,
                ));
            }
        }
        let mut row = try_join_all(row).await?;
        let mut mutation = self.mutation(row.remove(0).try_into()?);
        for (i, value) in row.iter().enumerate() {
            mutation.set(self.columns[i].0.clone(), value.clone().try_into()?)?;
        }

        self.chain
            .clone()
            .put(txn.clone(), mutation.key.clone(), mutation.values().into())
            .await?;

        Ok(().into())
    }
}

#[derive(Debug)]
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
        let mut columns: Vec<(String, Link)> = vec![];
        let mut key = None;

        for (name, datatype) in schema {
            if name == key_column {
                key = Some((name, datatype));
            } else {
                columns.push((name.clone(), datatype.clone()));
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

        let chain: Arc<Chain> = txn
            .clone()
            .post(
                &Link::to("/sbin/chain/new")?,
                vec![("path", txn.context().into())],
            )
            .await?
            .try_into()?;

        Ok(Arc::new(Table {
            key,
            columns,
            chain,
        }))
    }
}

#[async_trait]
impl TCExecutable for TableContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: &Link) -> TCResult<TCState> {
        if method != "/new" {
            return Err(error::bad_request(
                "TableContext has no such method",
                method,
            ));
        }

        let schema: Vec<(String, Link)> = txn.clone().require("schema")?.try_into()?;
        let key: String = txn.clone().require("key")?.try_into()?;
        Ok(TCState::Table(self.new_table(txn, schema, key).await?))
    }
}
