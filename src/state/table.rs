use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, try_join_all};
use futures::StreamExt;

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::TransactionCache;
use crate::internal::chain::{Chain, ChainBlock, Mutation};
use crate::internal::file::*;
use crate::state::schema::{Schema, SchemaHistory};
use crate::state::{Collection, Persistent, Transact};
use crate::transaction::{Transaction, TransactionId};
use crate::value::{PathSegment, TCResult, TCValue, ValueId};

type TableMutation = (Vec<TCValue>, Vec<Option<TCValue>>);

// TODO: rethink Mutation and Row data structures
impl Mutation for TableMutation {}

struct Row {
    key: Vec<TCValue>,
    values: Vec<Option<TCValue>>,
}

impl Row {
    fn update(&mut self, mutation: &TableMutation) {
        for (i, value) in mutation.1.iter().enumerate() {
            if !value.is_none() {
                self.values[i] = value.clone();
            }
        }
    }
}

impl From<Row> for TableMutation {
    fn from(row: Row) -> TableMutation {
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
    cache: TransactionCache<Vec<TCValue>, Vec<Option<TCValue>>>,
}

impl Table {
    async fn row_id(&self, txn: &Arc<Transaction>, value: &[TCValue]) -> TCResult<Vec<TCValue>> {
        let schema = self.schema.at(&txn.id()).await;
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
        let row_id = self.row_id(txn, row_id).await?;
        let schema = self.schema.at(&txn.id()).await;

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
            .stream_into::<TableMutation>(Some(txn.id()))
            .filter(|m: &TableMutation| future::ready(&m.0 == row_id))
            .fold(self.new_row(&txn, row_id).await?, |mut row, mutation| {
                row.update(&mutation);
                future::ready(row)
            })
            .await;

        if let Some(values) = self.cache.get(&txn.id(), row_id) {
            row.update(&(row_id.to_vec(), values));
        }

        Ok(row.values.iter().map(|v| v.into()).collect())
    }

    async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        row_id: Vec<TCValue>,
        column_values: Vec<TCValue>,
    ) -> TCResult<Arc<Self>> {
        let row_id = self.row_id(&txn, &row_id).await?;
        let schema = self.schema.at(&txn.id()).await;
        let schema_map = schema.as_map();

        let mut names = vec![];
        let mut values = vec![];
        for column_value in column_values.iter() {
            let (column, value): (ValueId, TCValue) = column_value.clone().try_into()?;

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

        let mut values: HashMap<ValueId, TCValue> = try_join_all(values)
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
        for (i, col) in schema.columns.iter().enumerate() {
            if let Some(value) = values.remove(&col.0) {
                mutated[i] = Some(value);
            }
        }

        let row = if let Some(values) = self.cache.get(&txn.id(), &row_id) {
            let mut row = Row {
                key: row_id.to_vec(),
                values,
            };
            row.update(&(row_id.to_vec(), mutated));
            row
        } else {
            Row {
                key: row_id.to_vec(),
                values: mutated,
            }
        };

        self.cache.insert(txn.id(), row_id, row.values);
        txn.mutate(self.clone());
        Ok(self.clone())
    }
}

#[async_trait]
impl File for Table {
    type Block = ChainBlock<TableMutation>;

    async fn copy_into(&self, txn_id: TransactionId, writer: &mut FileCopier) {
        println!("copying table into FileCopier");
        self.schema.copy_into(txn_id.clone(), writer).await;
        println!("copied schema");

        let schema = self.schema.at(&txn_id).await;
        println!("got current schema");
        let version: PathSegment = schema.version.to_string().try_into().unwrap();
        writer.write_file(
            version.try_into().unwrap(),
            Box::new(
                self.chain
                    .stream_bytes::<TableMutation>(Some(txn_id))
                    .boxed(),
            ),
        );
        println!("wrote table chain to file");
    }

    async fn copy_from(reader: &mut FileCopier, dest: Arc<Store>) -> Arc<Table> {
        println!("copying table from FileCopier");
        let schema_history = SchemaHistory::copy_from(reader, dest.clone()).await;
        println!("copied schema");

        println!("reading blocks from FileCopier");
        let (path, blocks) = reader.next().await.unwrap();
        println!("got blocks");
        let chain = Chain::copy_from(blocks, dest.reserve(path).unwrap()).await;
        println!("copied Chain");

        Arc::new(Table {
            schema: schema_history,
            chain,
            cache: TransactionCache::new(),
        })
    }

    async fn from_store(store: Arc<Store>) -> Arc<Table> {
        let schema =
            SchemaHistory::from_store(store.get_store(&"schema".try_into().unwrap()).unwrap())
                .await;

        let chain_path: PathSegment = schema
            .latest()
            .await
            .version
            .to_string()
            .try_into()
            .unwrap();
        let chain = Chain::from_store(store.get_store(&chain_path.try_into().unwrap()).unwrap())
            .await
            .unwrap();

        Arc::new(Table {
            schema,
            chain,
            cache: TransactionCache::new(),
        })
    }
}

#[async_trait]
impl Persistent for Table {
    type Config = Schema;

    async fn create(txn: Arc<Transaction>, schema: Schema) -> TCResult<Arc<Table>> {
        let table_chain = Chain::new(txn.context().reserve(schema.version.to_string())?);
        let schema_history = SchemaHistory::new(&txn, schema)?;

        Ok(Arc::new(Table {
            schema: schema_history,
            chain: table_chain,
            cache: TransactionCache::new(),
        }))
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TransactionId) {
        println!("Table::commit");
        let mutations: Vec<TableMutation> = self.cache.close(&txn_id).into_iter().collect();

        if !mutations.is_empty() {
            self.chain.put(&txn_id, &mutations).await
        }
        println!("Table::commit complete");
    }
}
