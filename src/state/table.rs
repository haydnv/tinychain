use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::Chain;
use crate::state::{Persistent, State};
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue, Version};

type Mutation = (Vec<TCValue>, Vec<Option<TCValue>>);

#[derive(Deserialize, Serialize)]
struct Schema {
    version: Version,
    key: Vec<(String, Link)>,
    columns: Vec<(String, Link)>,
}

impl Schema {
    fn as_map(self: &Arc<Self>) -> HashMap<String, Link> {
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

struct Row {
    key: Vec<TCValue>,
    values: Vec<Option<TCValue>>,
}

impl Row {
    fn update(&mut self, mutation: &Mutation) -> TCResult<()> {
        let (key, values) = mutation;
        if key != &self.key {
            let key: TCValue = key.clone().into();
            return Err(error::bad_request(
                "Cannot update the value of a row's primary key",
                key,
            ));
        }

        for (i, value) in values.iter().enumerate() {
            if !value.is_none() {
                self.values[i] = value.clone();
            }
        }

        Ok(())
    }
}

impl From<Row> for Mutation {
    fn from(row: Row) -> Mutation {
        (row.key, row.values)
    }
}

impl From<Row> for TCValue {
    fn from(row: Row) -> TCValue {
        let values: Vec<TCValue> = row
            .values
            .iter()
            .filter(|v| v.is_some())
            .map(|v| v.clone().unwrap())
            .collect();

        let mut value = Vec::with_capacity(row.key.len() + values.len());
        for k in row.key {
            value.push(k);
        }
        for v in values {
            value.push(v)
        }

        TCValue::Vector(value)
    }
}

pub struct Table {
    schema: Arc<Schema>,
    chain: Arc<Chain>,
    cache: RwLock<HashMap<TransactionId, Vec<Mutation>>>,
}

impl Table {
    async fn row_id(&self, txn: &Arc<Transaction>, value: &[TCValue]) -> TCResult<Vec<TCValue>> {
        let key_size = self.schema.key.len();

        let mut row_id: Vec<TCValue> = Vec::with_capacity(key_size);
        for value in try_join_all(
            value
                .iter()
                .enumerate()
                .map(|(i, v)| txn.get(&self.schema.key[i].1, v.clone())),
        )
        .await?
        {
            row_id.push(value.try_into()?)
        }
        Ok(row_id)
    }

    async fn new_row(&self, txn: &Arc<Transaction>, row_id: &[TCValue]) -> TCResult<Row> {
        let row_id = self.row_id(txn, row_id).await?;

        if row_id.len() != self.schema.key.len() {
            let key: TCValue = row_id.into();
            return Err(error::bad_request(
                &format!("Expected a key of length {}, found", self.schema.key.len()),
                key,
            ));
        }

        Ok(Row {
            key: row_id,
            values: iter::repeat(None).take(self.schema.columns.len()).collect(),
        })
    }
}

#[async_trait]
impl Persistent for Table {
    type Key = Vec<TCValue>;
    type Value = Vec<TCValue>;

    async fn commit(self: &Arc<Self>, txn_id: TransactionId) {
        let mutations = if let Some(mutations) = self.cache.read().unwrap().get(&txn_id) {
            mutations
                .iter()
                .map(|(k, v)| (k.clone().into(), v.clone().into()))
                .collect::<Vec<(TCValue, TCValue)>>()
        } else {
            vec![]
        };

        self.chain.put(txn_id, &mutations).await;
    }

    async fn get(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        row_id: &Self::Key,
    ) -> TCResult<Self::Value> {
        // TODO: use the TransactionId to get the state of the chain at a specific point in time

        let mut row = self.new_row(&txn, row_id).await?;
        let mutations: Vec<Mutation> = self.chain.get(txn.id(), &row.key.clone().into()).await?;
        for mutation in mutations {
            row.update(&mutation)?;
        }

        if let Some(mutations) = self.cache.read().unwrap().get(&txn.id()) {
            for mutation in mutations {
                row.update(mutation)?;
            }
        }

        Ok(row.values.iter().map(|o| o.into()).collect())
    }

    async fn put(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        row_id: Self::Key,
        column_values: Self::Value,
    ) -> TCResult<Arc<Self>> {
        let row_id = self.row_id(&txn, &row_id).await?;
        let schema: HashMap<String, Link> = self.schema.as_map();

        let mut names = vec![];
        let mut values = vec![];
        for column_value in column_values.iter() {
            let (column, value): (String, TCValue) = column_value.clone().try_into()?;

            if let Some(ctr) = schema.get(&column) {
                names.push(column);
                values.push(txn.get(ctr, value));
            } else {
                return Err(error::bad_request(
                    "This table contains no such column",
                    column,
                ));
            }
        }

        let mut values: HashMap<String, TCValue> = try_join_all(values)
            .await?
            .iter()
            .map(|v| v.clone().try_into())
            .collect::<TCResult<Vec<TCValue>>>()?
            .iter()
            .enumerate()
            .map(|(i, v)| (names[i].clone(), v.clone()))
            .collect();

        let mut mutated: Vec<Option<TCValue>> =
            iter::repeat(None).take(self.schema.columns.len()).collect();
        for i in 0..self.schema.columns.len() {
            if let Some(value) = values.remove(&self.schema.columns[i].0) {
                mutated[i] = Some(value);
            }
        }

        let mutation = (row_id, mutated);

        let mut cache = self.cache.write().unwrap();
        if let Some(mutations) = cache.get_mut(&txn.id()) {
            mutations.push(mutation);
        } else {
            cache.insert(txn.id(), vec![mutation]);
        }

        Ok(self.clone().into())
    }
}

#[derive(Debug)]
pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
    }

    pub async fn new_table(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        method: &Link,
    ) -> TCResult<State> {
        if method != "/new" {
            return Err(error::bad_request(
                "TableContext has no such method",
                method,
            ));
        }

        let key: Vec<(String, Link)> = txn.require("key")?.try_into()?;
        let columns: Vec<(String, Link)> = txn.require("columns")?.try_into()?;

        let schema = Arc::new(Schema {
            key,
            columns,
            version: Version::parse("1.0.0")?,
        });

        let table_chain = Chain::new(
            txn.context()
                .reserve(&Link::to(&format!("/{}", schema.version))?)?,
        );
        Ok(Arc::new(Table {
            schema,
            chain: Arc::new(table_chain),
            cache: RwLock::new(HashMap::new()),
        })
        .into())
    }
}
