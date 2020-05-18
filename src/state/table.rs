use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, try_join_all};
use futures::lock::Mutex;
use futures::StreamExt;

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{Chain, ChainBlock, Mutation};
use crate::internal::file::*;
use crate::object::actor::Token;
use crate::state::schema::{Schema, SchemaHistory};
use crate::state::{Collection, Persistent};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{PathSegment, TCResult, TCValue, ValueId};

type Row = (Vec<TCValue>, Vec<Option<TCValue>>);

impl Mutation for Row {}

fn update_row(row: &mut Row, mut values: Vec<Option<TCValue>>) {
    let mut i = values.len();
    while !values.is_empty() {
        if let Some(value) = values.pop() {
            row.1[i - 1] = value;
        }
        i -= 1;
    }
}

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

        Ok((
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
        let row = self
            .chain
            .lock()
            .await
            .stream_into(txn.id())
            .filter(|r: &Row| future::ready(&r.0 == row_id))
            .fold(
                self.new_row(&txn, row_id, auth).await?,
                |mut row, mutation| {
                    update_row(&mut row, mutation.1);
                    future::ready(row)
                },
            )
            .await;

        Ok(row.1.into_iter().map(|v| v.into()).collect())
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

        let row: Row = (row_id, mutated);
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
    type Block = ChainBlock<Row>;

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
        let table_chain = Mutex::new(
            Chain::new(
                &txn.id(),
                txn.context()
                    .reserve(&txn.id(), schema.version.to_string().parse()?)
                    .await?,
            )
            .await,
        );
        let schema_history = SchemaHistory::new(&txn, schema).await?;

        Ok(Arc::new(Table {
            schema: schema_history,
            chain: table_chain,
        }))
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        self.chain.lock().await.commit(txn_id).await
    }
}
