use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, try_join_all};
use futures::lock::Mutex;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::auth::Token;
use crate::error;
use crate::internal::chain::{Chain, Mutation};
use crate::internal::file::*;
use crate::internal::{Dir, History};
use crate::state::{Args, Collection, Persistent, State};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{PathSegment, TCPath};
use crate::value::{TCResult, Value, ValueId, Version};

#[derive(Clone, Deserialize, Serialize)]
pub struct Schema {
    key: Vec<(ValueId, TCPath)>,
    columns: Vec<(ValueId, TCPath)>,
    version: Version,
}

impl Schema {
    fn as_map(&self) -> HashMap<ValueId, TCPath> {
        [&self.key[..], &self.columns[..]]
            .concat()
            .into_iter()
            .collect()
    }
}

impl Default for Schema {
    fn default() -> Schema {
        Schema {
            key: vec![],
            columns: vec![],
            version: "0.0.0".parse().unwrap(),
        }
    }
}

impl Mutation for Schema {}

impl TryFrom<Args> for Schema {
    type Error = error::TCError;

    fn try_from(mut args: Args) -> TCResult<Schema> {
        let key: Vec<(ValueId, TCPath)> = args.take("key")?;
        let columns: Vec<(ValueId, TCPath)> = args.take("columns")?;
        let version: Version = args.take("version")?;

        args.assert_empty()?;
        Ok(Schema {
            key,
            columns,
            version,
        })
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Row(Vec<Value>, Vec<Option<Value>>);

impl From<(Vec<Value>, Vec<Option<Value>>)> for Row {
    fn from(data: (Vec<Value>, Vec<Option<Value>>)) -> Row {
        Row(data.0, data.1)
    }
}

impl From<Row> for (Vec<Value>, Vec<Option<Value>>) {
    fn from(row: Row) -> (Vec<Value>, Vec<Option<Value>>) {
        (row.0, row.1)
    }
}

impl From<Row> for Value {
    fn from(mut row: Row) -> Value {
        Value::Vector(vec![
            row.0.into(),
            row.1.drain(..).map(|i| i.unwrap_or(Value::None)).collect(),
        ])
    }
}

impl TryFrom<Value> for Row {
    type Error = error::TCError;

    fn try_from(value: Value) -> TCResult<Row> {
        let mut value: Vec<Value> = value.try_into()?;
        if value.len() == 2 {
            let mut row_values: Vec<Value> = value.pop().unwrap().try_into()?;
            let row_values: Vec<Option<Value>> = row_values.drain(..).map(|v| v.into()).collect();
            let row_key: Vec<Value> = value.pop().unwrap().try_into()?;
            Ok(Row(row_key, row_values))
        } else {
            let value: Value = value.into();
            Err(error::bad_request("Expected Row but found", value))
        }
    }
}

impl Mutation for Row {}

pub struct Table {
    schema_history: Arc<History<Schema>>,
    chain: Mutex<Chain<Row>>,
}

impl Table {
    async fn schema(&self, txn_id: TxnId) -> TCResult<Schema> {
        self.schema_history
            .at(txn_id)
            .await
            .ok_or_else(|| error::internal("This Table has no Schema!"))
    }

    async fn row_id(
        &self,
        txn: &Arc<Txn<'_>>,
        value: &[Value],
        auth: &Option<Token>,
    ) -> TCResult<Vec<Value>> {
        let schema = self.schema(txn.id()).await?;
        let key_size = schema.key.len();

        if value.len() != key_size {
            return Err(error::bad_request(
                &format!("Expected a key of length {}, found", key_size),
                value.len(),
            ));
        }

        let mut row_id: Vec<Value> = Vec::with_capacity(key_size);
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
        row_id: &[Value],
        auth: &Option<Token>,
    ) -> TCResult<Row> {
        let row_id = self.row_id(txn, row_id, auth).await?;
        let schema = self.schema(txn.id()).await?;

        if row_id.len() != schema.key.len() {
            let key: Value = row_id.into();
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
    type Key = Vec<Value>;
    type Value = Vec<Value>;

    async fn get(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        row_id: &Self::Key,
    ) -> TCResult<Self::Value> {
        // TODO: authorize
        let auth = &None;
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

        Ok(row.1.drain(..).map(|v| v.unwrap_or(Value::None)).collect())
    }

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        row_id: Vec<Value>,
        column_values: Vec<Value>,
    ) -> TCResult<State> {
        // TODO: authorize
        let auth = &None;
        let row_id = self.row_id(&txn, &row_id, auth).await?;
        let schema = self.schema(txn.id()).await?;
        let schema_map = schema.as_map();

        let mut names = vec![];
        let mut values = vec![];
        for column_value in column_values.iter() {
            let (column, value): (ValueId, Value) = column_value.clone().try_into()?;

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

        let mut values: HashMap<ValueId, Value> = try_join_all(values)
            .await?
            .iter()
            .map(|v| v.clone().try_into())
            .collect::<TCResult<Vec<Value>>>()?
            .iter()
            .enumerate()
            .map(|(i, v)| (names[i].clone(), v.clone()))
            .collect();

        let mut mutated: Vec<Option<Value>> =
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
        Ok(self.into())
    }
}

#[async_trait]
impl File for Table {
    async fn copy_into(&self, txn_id: TxnId, writer: &mut FileCopier) {
        println!("copying table into FileCopier");
        self.schema_history.copy_into(txn_id.clone(), writer).await;
        println!("copied schema");

        let schema = self.schema(txn_id.clone()).await.unwrap();
        println!("got current schema");
        let version: PathSegment = schema.version.to_string().parse().unwrap();
        writer.write_file(
            version.try_into().unwrap(),
            Box::new(self.chain.lock().await.stream_bytes(txn_id).boxed()),
        );
        println!("wrote table chain to file");
    }

    async fn copy_from(reader: &mut FileCopier, txn_id: &TxnId, dest: Arc<Dir>) -> Arc<Table> {
        println!("copying table from FileCopier");
        let schema_history = History::copy_from(
            reader,
            txn_id,
            dest.create_dir(txn_id, "history".parse().unwrap())
                .await
                .unwrap()
                .clone(),
        )
        .await;
        println!("copied schema");

        println!("reading blocks from FileCopier");
        let (path, blocks) = reader.next().await.unwrap();
        println!("got blocks");
        let chain = Mutex::new(
            Chain::copy_from(
                blocks,
                txn_id.clone(),
                dest.create_store(txn_id, path.try_into().unwrap())
                    .await
                    .unwrap(),
            )
            .await,
        );
        println!("copied Chain");

        Arc::new(Table {
            schema_history,
            chain,
        })
    }

    async fn from_dir(txn_id: &TxnId, dir: Arc<Dir>) -> Arc<Table> {
        let schema_history: Arc<History<Schema>> = History::from_dir(
            txn_id,
            dir.get_dir(txn_id, &"history".parse().unwrap())
                .await
                .unwrap()
                .unwrap(),
        )
        .await;

        let latest_schema = schema_history.at(txn_id.clone()).await.unwrap();
        let chain_path: PathSegment = latest_schema.version.to_string().parse().unwrap();
        let chain = Mutex::new(
            Chain::from_store(
                txn_id,
                dir.get_store(txn_id, &chain_path.try_into().unwrap())
                    .await
                    .unwrap(),
            )
            .await
            .unwrap(),
        );

        Arc::new(Table {
            schema_history,
            chain,
        })
    }
}

#[async_trait]
impl Persistent for Table {
    type Config = Schema;

    async fn create(txn: &Arc<Txn<'_>>, schema: Schema) -> TCResult<Arc<Table>> {
        let txn_id = &txn.id();

        let chain = Chain::new(
            txn.id(),
            txn.context()
                .create_store(txn_id, schema.version.to_string().parse()?)
                .await?,
        )
        .await;

        let schema_history = History::new(
            txn.id(),
            txn.context().create_dir(txn_id, "history".parse()?).await?,
        )
        .await?;
        schema_history.put(txn.id(), schema).await?;
        txn.mutate(schema_history.clone());

        Ok(Arc::new(Table {
            schema_history,
            chain: Mutex::new(chain),
        }))
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        self.chain.lock().await.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.chain.lock().await.commit(txn_id).await
    }
}
