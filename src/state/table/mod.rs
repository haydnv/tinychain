use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::future::{try_join, try_join_all};
use futures::StreamExt;

use crate::error;
use crate::internal::file::*;
use crate::internal::{Chain, FsDir};
use crate::state::{Collection, Persistent};
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue, Version};

mod schema;
use schema::{Schema, SchemaHistory};

type Mutation = (Vec<TCValue>, Vec<Option<TCValue>>);

struct Row {
    key: Vec<TCValue>,
    values: Vec<Option<TCValue>>,
}

impl Row {
    fn update(&mut self, mutation: &Mutation) {
        for (i, value) in mutation.1.iter().enumerate() {
            if !value.is_none() {
                self.values[i] = value.clone();
            }
        }
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
    schema: Arc<SchemaHistory>,
    chain: Arc<Chain>,
    cache: RwLock<HashMap<TransactionId, Vec<Mutation>>>,
}

impl Table {
    async fn row_id(&self, txn: &Arc<Transaction>, value: &[TCValue]) -> TCResult<Vec<TCValue>> {
        let schema = self.schema.current(txn.id()).await?;
        let key_size = schema.key.len();

        let mut row_id: Vec<TCValue> = Vec::with_capacity(key_size);
        for value in try_join_all(
            value
                .iter()
                .enumerate()
                .map(|(i, v)| txn.get(&schema.key[i].1, v.clone())),
        )
        .await?
        {
            row_id.push(value.try_into()?)
        }
        Ok(row_id)
    }

    async fn new_row(&self, txn: &Arc<Transaction>, row_id: &[TCValue]) -> TCResult<Row> {
        let (row_id, schema) =
            try_join(self.row_id(txn, row_id), self.schema.current(txn.id())).await?;

        if row_id.len() != schema.key.len() {
            let key: TCValue = row_id.into();
            return Err(error::bad_request(
                &format!("Expected a key of length {}, found", schema.key.len()),
                key,
            ));
        }

        Ok(Row {
            key: row_id,
            values: iter::repeat(None).take(schema.columns.len()).collect(),
        })
    }
}

#[async_trait]
impl Collection for Table {
    type Key = Vec<TCValue>;
    type Value = Vec<TCValue>;

    async fn get(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        row_id: &Self::Key,
    ) -> TCResult<Self::Value> {
        let mut row = self
            .chain
            .until(txn.id())
            .fold(
                self.new_row(&txn, row_id).await?,
                |mut row, block: Vec<Mutation>| async {
                    for mutation in block {
                        if mutation.0 == row.key {
                            row.update(&mutation)
                        }
                    }
                    row
                },
            )
            .await;

        if let Some(mutations) = self.cache.read().unwrap().get(&txn.id()) {
            for mutation in mutations {
                row.update(mutation);
            }
        }

        Ok(row.values.iter().map(|v| v.into()).collect())
    }

    async fn put(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        row_id: Self::Key,
        column_values: Self::Value,
    ) -> TCResult<Arc<Self>> {
        let (row_id, schema) =
            try_join(self.row_id(&txn, &row_id), self.schema.current(txn.id())).await?;
        let schema_map = schema.as_map();

        let mut names = vec![];
        let mut values = vec![];
        for column_value in column_values.iter() {
            let (column, value): (String, TCValue) = column_value.clone().try_into()?;

            if let Some(ctr) = schema_map.get(&column) {
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
            iter::repeat(None).take(schema.columns.len()).collect();
        for i in 0..schema.columns.len() {
            if let Some(value) = values.remove(&schema.columns[i].0) {
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

        Ok(self.clone())
    }
}

#[async_trait]
impl File for Table {
    async fn copy_from(reader: &mut FileReader, dest: Arc<FsDir>) -> TCResult<Arc<Table>> {
        let schema_history = SchemaHistory::copy_from(reader, dest.clone()).await?;

        let (path, blocks) = reader.next().await.unwrap();
        let chain = Chain::from(blocks, dest.reserve(&path)?).await;

        Ok(Arc::new(Table {
            schema: schema_history,
            chain,
            cache: RwLock::new(HashMap::new()),
        }))
    }

    async fn copy_to(&self, txn_id: TransactionId, writer: &mut FileWriter) -> TCResult<()> {
        self.schema.copy_to(txn_id.clone(), writer).await?;

        let schema = self.schema.current(txn_id.clone()).await?;
        let chain_path = format!("/{}", schema.version);
        writer.write_file(
            Link::to(&chain_path).unwrap(),
            Box::new(self.chain.into_stream(txn_id).boxed()),
        );

        Ok(())
    }
}

#[async_trait]
impl Persistent for Table {
    async fn commit(&self, txn_id: TransactionId) {
        let mutations = if let Some(mutations) = self.cache.read().unwrap().get(&txn_id) {
            mutations
                .iter()
                .map(|(k, v)| (k.clone().into(), v.clone().into()))
                .collect::<Vec<(TCValue, TCValue)>>()
        } else {
            vec![]
        };

        self.chain.clone().put(&txn_id, &mutations).await;
    }
}

#[derive(Debug)]
pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
    }

    pub async fn new_table(self: &Arc<Self>, txn: Arc<Transaction>) -> TCResult<Arc<Table>> {
        let key: Vec<(String, Link)> = txn.require("key")?.try_into()?;
        let columns: Vec<(String, Link)> = txn.require("columns")?.try_into()?;

        let schema = Schema {
            key,
            columns,
            version: Version::parse("1.0.0")?,
        };

        let table_chain = Chain::new(
            txn.context()
                .reserve(&Link::to(&format!("/{}", schema.version))?)?,
        );
        let schema_history = SchemaHistory::new(&txn, schema)?;

        Ok(Arc::new(Table {
            schema: schema_history,
            chain: table_chain,
            cache: RwLock::new(HashMap::new()),
        }))
    }
}
