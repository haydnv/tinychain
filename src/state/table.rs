use std::collections::HashMap;
use std::convert::TryInto;
use std::hash;
use std::iter::{repeat, FromIterator};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::future::try_join_all;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::internal::Chain;
use crate::state::TCState;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCValue};

#[derive(Clone)]
struct Mutation {
    schema: HashMap<String, usize>,
    key: TCValue,
    values: Vec<Option<TCValue>>,
}

impl hash::Hash for Mutation {
    fn hash<H: hash::Hasher>(&self, h: &mut H) {
        self.key.hash(h);
        self.values.hash(h);
    }
}

impl Mutation {
    fn apply(&mut self, values: &[Option<TCValue>]) {
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

pub struct Table {
    key: (String, Link),
    columns: Vec<(String, Link)>,
    chain: Arc<Chain>,
    cache: RwLock<HashMap<TransactionId, Vec<Mutation>>>,
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

impl hash::Hash for Table {
    fn hash<H: hash::Hasher>(&self, h: &mut H) {
        self.key.hash(h);
        self.columns.hash(h);
    }
}

#[async_trait]
impl TCContext for Table {
    async fn commit(self: &Arc<Self>, txn_id: TransactionId) {
        let mutations = if let Some(mutations) = self.cache.read().unwrap().get(&txn_id) {
            mutations
                .iter()
                .map(|m| (m.key.clone(), m.values()))
                .collect()
        } else {
            vec![]
        };

        self.chain.put(txn_id, mutations).await;
    }

    async fn get(self: &Arc<Self>, txn: Arc<Transaction>, row_id: &TCValue) -> TCResult<TCState> {
        // TODO: use the TransactionId to get the state of the chain at a specific point in time

        let mut row = self.mutation(row_id.clone());
        let mutations: Vec<TCValue> = self.chain.get(txn.id(), row_id).await?.try_into()?;
        for mutation in mutations {
            let mutation: Vec<Option<TCValue>> = mutation.try_into()?;
            row.apply(&mutation);
        }

        if let Some(mutations) = self.cache.read().unwrap().get(&txn.id()) {
            for mutation in mutations {
                row.apply(&mutation.values);
            }
        }

        Ok(row.values().into())
    }

    async fn put(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        row_id: TCValue,
        column_values: TCState,
    ) -> TCResult<TCState> {
        let column_values: Vec<TCValue> = column_values.try_into()?;
        let schema = self.schema_map();

        let mut row = vec![];
        row.push(txn.post(&self.key.1, vec![("from".into(), row_id)]));
        for column_value in column_values.iter() {
            let column_value: (String, TCValue) = column_value.clone().try_into()?;
            let (column, value) = column_value;

            if let Some(ctr) = schema.get(&column) {
                row.push(txn.post(ctr, vec![("from".into(), value)]));
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

        let mut cache = self.cache.write().unwrap();
        if let Some(mutations) = cache.get_mut(&txn.id()) {
            mutations.push(mutation);
        } else {
            cache.insert(txn.id(), vec![mutation]);
        }

        Ok(self.into())
    }
}

#[derive(Debug)]
pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
    }

    async fn new_table<'a>(
        self: &Arc<Self>,
        fs_dir: Arc<fs::Dir>,
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

        let chain = Chain::new(fs_dir.reserve(&Link::to("/chain")?)?);

        Ok(Arc::new(Table {
            key,
            columns,
            chain,
            cache: RwLock::new(HashMap::new()),
        }))
    }
}

#[async_trait]
impl TCExecutable for TableContext {
    async fn post(self: &Arc<Self>, txn: Arc<Transaction>, method: &Link) -> TCResult<TCState> {
        if method != "/new" {
            return Err(error::bad_request(
                "TableContext has no such method",
                method,
            ));
        }

        let schema: Vec<(String, Link)> = txn.clone().require("schema")?.try_into()?;
        let key: String = txn.clone().require("key")?.try_into()?;
        Ok(TCState::Table(
            self.new_table(txn.context(), schema, key).await?,
        ))
    }
}
