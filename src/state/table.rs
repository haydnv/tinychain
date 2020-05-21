use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, try_join_all};
use futures::lock::Mutex;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{Chain, Mutation};
use crate::internal::file::*;
use crate::object::actor::Token;
use crate::state::SchemaHistory;
use crate::state::{Collection, Persistent};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{PathSegment, TCPath};
use crate::value::{TCResult, TCValue, ValueId, Version};

#[derive(Clone, Deserialize, Serialize)]
pub struct Schema {
    pub key: Vec<(ValueId, TCPath)>,
    pub columns: Vec<(ValueId, TCPath)>,
    pub version: Version,
}

impl Schema {
    pub fn as_map(&self) -> HashMap<ValueId, TCPath> {
        [&self.key[..], &self.columns[..]]
            .concat()
            .into_iter()
            .collect()
    }

    pub fn from(
        key: Vec<(ValueId, TCPath)>,
        columns: Vec<(ValueId, TCPath)>,
        version: Version,
    ) -> Schema {
        Schema {
            key,
            columns,
            version,
        }
    }

    pub fn new() -> Schema {
        Schema {
            key: vec![],
            columns: vec![],
            version: "0.0.0".parse().unwrap(),
        }
    }
}

impl Mutation for Schema {}

impl TryFrom<TCValue> for Schema {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Schema> {
        let value: Vec<TCValue> = value.try_into()?;
        if value.len() == 3 {
            let key: Vec<(ValueId, TCPath)> = value[0].clone().try_into()?;
            let columns: Vec<(ValueId, TCPath)> = value[1].clone().try_into()?;
            let version: Version = value[2].clone().try_into()?;
            Ok(Schema {
                key,
                columns,
                version,
            })
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Schema of the form ([(name, constructor)...], [(name, constructor), ...], Version), found", value))
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Row(Vec<TCValue>, Vec<Option<TCValue>>);

impl Row {
    pub fn from(key: Vec<TCValue>, values: Vec<Option<TCValue>>) -> Row {
        Row(key, values)
    }
}

impl From<Row> for (Vec<TCValue>, Vec<Option<TCValue>>) {
    fn from(row: Row) -> (Vec<TCValue>, Vec<Option<TCValue>>) {
        (row.0, row.1)
    }
}

impl From<Row> for TCValue {
    fn from(mut row: Row) -> TCValue {
        TCValue::Vector(vec![
            row.0.into(),
            row.1
                .drain(..)
                .map(|i| i.unwrap_or(TCValue::None))
                .collect(),
        ])
    }
}

impl TryFrom<TCValue> for Row {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Row> {
        let mut value: Vec<TCValue> = value.try_into()?;
        if value.len() == 2 {
            let mut row_values: Vec<TCValue> = value.pop().unwrap().try_into()?;
            let row_values: Vec<Option<TCValue>> = row_values.drain(..).map(|v| v.into()).collect();
            let row_key: Vec<TCValue> = value.pop().unwrap().try_into()?;
            Ok(Row(row_key, row_values))
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Row but found", value))
        }
    }
}

impl Mutation for Row {}

pub struct Table {
    schema: Arc<SchemaHistory>,
    chain: Mutex<Chain<Row>>,
}

impl Table {
    async fn row_id(
        &self,
        txn: &Arc<Txn<'_>>,
        value: &[TCValue],
        auth: &Option<Token>,
    ) -> TCResult<Vec<TCValue>> {
        let schema = self.schema.at(txn.id()).await;
        let key_size = schema.key.len();

        let mut row_id: Vec<TCValue> = Vec::with_capacity(key_size);
        for value in try_join_all(
            value
                .iter()
                .enumerate()
                .map(|(i, v)| txn.get(schema.key[i].1.clone().into(), v.clone(), auth)),
        )
        .await?
        {
            row_id.push(value.try_into()?)
        }

        Ok(row_id)
    }

    async fn new_row(
        &self,
        txn: &Arc<Txn<'_>>,
        row_id: &[TCValue],
        auth: &Option<Token>,
    ) -> TCResult<Row> {
        let row_id = self.row_id(txn, row_id, auth).await?;
        let schema = self.schema.at(txn.id()).await;

        if row_id.len() != schema.key.len() {
            let key: TCValue = row_id.into();
            return Err(error::bad_request(
                &format!("Expected a key of length {}, found", schema.key.len()),
                key,
            ));
        }

        Ok(Row(
            row_id,
            iter::repeat(None).take(schema.columns.len()).collect(),
        ))
    }
}

#[async_trait]
impl Collection for Table {
    type Key = Vec<TCValue>;
    type Value = Vec<TCValue>;

    async fn get(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        row_id: &Self::Key,
        auth: &Option<Token>,
    ) -> TCResult<Self::Value> {
        let mut row = self
            .chain
            .lock()
            .await
            .stream_into(txn.id())
            .filter(|r: &Row| future::ready(&r.0 == row_id))
            .fold(
                self.new_row(&txn, row_id, auth).await?,
                |mut row, mut mutation| {
                    for (i, value) in mutation.1.drain(..).enumerate() {
                        if let Some(value) = value {
                            row.1[i] = Some(value);
                        }
                    }

                    future::ready(row)
                },
            )
            .await;

        Ok(row
            .1
            .drain(..)
            .map(|v| v.unwrap_or(TCValue::None))
            .collect())
    }

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        row_id: Vec<TCValue>,
        column_values: Vec<TCValue>,
        auth: &Option<Token>,
    ) -> TCResult<Arc<Self>> {
        let row_id = self.row_id(&txn, &row_id, auth).await?;
        let schema = self.schema.at(txn.id()).await;
        let schema_map = schema.as_map();

        let mut names = vec![];
        let mut values = vec![];
        for column_value in column_values.iter() {
            let (column, value): (ValueId, TCValue) = column_value.clone().try_into()?;

            if let Some(ctr) = schema_map.get(&column) {
                names.push(column);
                values.push(txn.get(ctr.clone().into(), value, auth));
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

        let row = Row(row_id, mutated);

        self.chain
            .lock()
            .await
            .put(txn.id(), iter::once(row))
            .await?;
        txn.mutate(self.clone());
        Ok(self.clone())
    }
}

#[async_trait]
impl File for Table {
    async fn copy_into(&self, txn_id: TxnId, writer: &mut FileCopier) {
        println!("copying table into FileCopier");
        self.schema.copy_into(txn_id.clone(), writer).await;
        println!("copied schema");

        let schema = self.schema.at(txn_id.clone()).await;
        println!("got current schema");
        let version: PathSegment = schema.version.to_string().parse().unwrap();
        writer.write_file(
            version.try_into().unwrap(),
            Box::new(self.chain.lock().await.stream_bytes(txn_id).boxed()),
        );
        println!("wrote table chain to file");
    }

    async fn copy_from(reader: &mut FileCopier, txn_id: &TxnId, dest: Arc<Store>) -> Arc<Table> {
        println!("copying table from FileCopier");
        let schema_history = SchemaHistory::copy_from(reader, txn_id, dest.clone()).await;
        println!("copied schema");

        println!("reading blocks from FileCopier");
        let (path, blocks) = reader.next().await.unwrap();
        println!("got blocks");
        let chain = Mutex::new(
            Chain::copy_from(blocks, txn_id, dest.reserve(txn_id, path).await.unwrap()).await,
        );
        println!("copied Chain");

        Arc::new(Table {
            schema: schema_history,
            chain,
        })
    }

    async fn from_store(txn_id: &TxnId, store: Arc<Store>) -> Arc<Table> {
        let schema = SchemaHistory::from_store(
            txn_id,
            store
                .get_store(txn_id, &"schema".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        let chain_path: PathSegment = schema
            .at(txn_id.clone())
            .await
            .version
            .to_string()
            .parse()
            .unwrap();
        let chain = Mutex::new(
            Chain::from_store(
                txn_id,
                store
                    .get_store(txn_id, &chain_path.try_into().unwrap())
                    .await
                    .unwrap(),
            )
            .await
            .unwrap(),
        );

        Arc::new(Table { schema, chain })
    }
}

#[async_trait]
impl Persistent for Table {
    type Config = Schema;

    async fn create(txn: &Arc<Txn<'_>>, schema: Schema) -> TCResult<Arc<Table>> {
        let chain = Chain::new(
            &txn.id(),
            txn.context()
                .reserve(&txn.id(), schema.version.to_string().parse()?)
                .await?,
        )
        .await;

        let schema_history = SchemaHistory::new(&txn, schema).await?;

        Ok(Arc::new(Table {
            schema: schema_history,
            chain: Mutex::new(chain),
        }))
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        self.chain.lock().await.commit(txn_id).await
    }
}
