use std::collections::BTreeMap;
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::stream::{self, StreamExt};
use futures::{join, try_join, TryFutureExt, TryStreamExt};
use log::{debug, error};
use safecast::*;

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableInstance;
#[cfg(feature = "tensor")]
use tc_tensor::TensorAccess;
use tc_transact::fs::*;
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::{label, Id, Instance, Label, Map, NativeClass, TCPathBuf, TCTryStream, Tuple};

use crate::chain::{ChainType, Subject, CHAIN, NULL_HASH};
use crate::collection::*;
use crate::fs;
use crate::route::Public;
use crate::scalar::{OpRef, Scalar, TCRef, Value};
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{ChainBlock, Mutation};

const DATA: Label = label("data");

#[derive(Clone)]
pub struct History {
    dir: fs::Dir,
    file: fs::File<ChainBlock>,
    latest: TxnLock<Mutable<u64>>,
}

impl History {
    fn new(latest: u64, dir: fs::Dir, file: fs::File<ChainBlock>) -> Self {
        let latest = TxnLock::new("latest block ordinal", latest.into());
        Self { dir, latest, file }
    }

    pub async fn create(txn_id: TxnId, dir: fs::Dir, class: ChainType) -> TCResult<Self> {
        let file: fs::File<ChainBlock> = dir.create_file(txn_id, CHAIN.into(), class).await?;
        file.create_block(txn_id, 0u64.into(), ChainBlock::new(NULL_HASH))
            .await?;

        let dir = dir.create_dir(txn_id, DATA.into()).await?;

        Ok(Self::new(0, dir, file))
    }

    pub async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        debug!("History::append_delete {} {} {}", txn_id, path, key);
        let mut block = self.write_latest(txn_id).await?;
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    pub async fn append_put(
        &self,
        txn: Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        let txn_id = *txn.id();
        let value = self.save_state(txn, value).await?;

        debug!(
            "History::append_put {} {} {:?} {:?}",
            txn_id, path, key, value
        );

        let mut block = self.write_latest(txn_id).await?;
        block.append_put(txn_id, path, key, value);

        Ok(())
    }

    async fn save_state(&self, txn: Txn, state: State) -> TCResult<Scalar> {
        if state.is_ref() {
            return Err(TCError::bad_request(
                "cannot update Chain with reference: {}",
                state,
            ));
        }

        let txn_id = *txn.id();
        match state {
            State::Collection(collection) => match collection {
                Collection::BTree(btree) => {
                    let hash: Id = btree.hash_hex(&txn).await?.parse()?;
                    let schema = btree.schema().to_vec();
                    let classpath = BTreeType::default().path();

                    if self.dir.contains(&txn_id, &hash).await? {
                        debug!("BTree with hash {} is already saved", hash);
                    } else {
                        let file = self
                            .dir
                            .create_file(txn_id, hash.clone(), btree.class())
                            .await?;

                        BTreeFile::copy_from(btree, file, txn).await?;
                        debug!("saved BTree with hash {}", hash);
                    }

                    Ok(OpRef::Get((
                        (hash.into(), classpath).into(),
                        Value::from_iter(schema).into(),
                    ))
                    .into())
                }
                Collection::Table(table) => {
                    let hash: Id = table.hash_hex(&txn).await?.parse()?;
                    let schema = table.schema().clone();
                    let classpath = TableType::default().path();

                    if self.dir.contains(&txn_id, &hash).await? {
                        debug!("Table with hash {} is already saved", hash);
                    } else {
                        let dir = self.dir.create_dir(txn_id, hash.clone()).await?;
                        TableIndex::copy_from(table, dir, txn).await?;
                        debug!("saved Table with hash {}", hash);
                    }

                    Ok(OpRef::Get((
                        (hash.into(), classpath).into(),
                        Value::cast_from(schema).into(),
                    ))
                    .into())
                }

                #[cfg(feature = "tensor")]
                Collection::Tensor(tensor) => {
                    let schema = cast_tensor_schema(&tensor);
                    let classpath = tensor.class().path();

                    let hash = match tensor {
                        Tensor::Dense(dense) => {
                            let hash = dense.hash_hex(&txn).await?.parse()?;

                            if self.dir.contains(&txn_id, &hash).await? {
                                debug!("Tensor with hash {} is already saved", hash);
                            } else {
                                let file = self
                                    .dir
                                    .create_file(txn_id, hash.clone(), TensorType::Dense)
                                    .await?;

                                DenseTensor::copy_from(dense, file, txn).await?;
                                debug!("saved Tensor with hash {}", hash);
                            }

                            hash
                        }
                        Tensor::Sparse(sparse) => {
                            let hash = sparse.hash_hex(&txn).await?.parse()?;

                            if self.dir.contains(&txn_id, &hash).await? {
                                debug!("Tensor with hash {} is already saved", hash);
                            } else {
                                let dir = self.dir.create_dir(txn_id, hash.clone()).await?;
                                SparseTensor::copy_from(sparse, dir, txn).await?;
                                debug!("saved Tensor with hash {}", hash);
                            }

                            hash
                        }
                    };

                    Ok(OpRef::Get(((hash.into(), classpath).into(), schema.into())).into())
                }
            },
            State::Scalar(value) => Ok(value),
            other if Scalar::can_cast_from(&other) => Ok(other.opt_cast_into().unwrap()),
            other => Err(TCError::bad_request(
                "Chain does not support value",
                other.class(),
            )),
        }
    }

    pub async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        let block = self.read_latest(txn_id).await?;
        Ok(block.mutations().keys().next().cloned())
    }

    pub async fn latest_block_id(&self, txn_id: &TxnId) -> TCResult<u64> {
        self.latest.read(txn_id).map_ok(|id| *id).await
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: u64) -> TCResult<bool> {
        self.file.contains_block(txn_id, &block_id.into()).await
    }

    pub async fn create_next_block(&self, txn_id: TxnId) -> TCResult<fs::Block<ChainBlock>> {
        let mut latest = self.latest.write(txn_id).await?;
        let last_block = self.read_block(txn_id, (*latest).into()).await?;
        let hash = last_block.hash().await?;
        let block = ChainBlock::new(hash);

        (*latest) += 1;
        debug!("creating next chain block {}", *latest);

        self.file
            .create_block(txn_id, (*latest).into(), block)
            .await
    }

    pub async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: u64,
    ) -> TCResult<fs::BlockRead<ChainBlock>> {
        self.file.read_block(txn_id, block_id.into()).await
    }

    pub async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: u64,
    ) -> TCResult<fs::BlockWrite<ChainBlock>> {
        self.file.write_block(txn_id, block_id.into()).await
    }

    pub async fn read_latest(&self, txn_id: TxnId) -> TCResult<fs::BlockRead<ChainBlock>> {
        let latest = self.latest.read(&txn_id).await?;
        self.read_block(txn_id, (*latest).into()).await
    }

    pub async fn write_latest(&self, txn_id: TxnId) -> TCResult<fs::BlockWrite<ChainBlock>> {
        let latest = self.latest.read(&txn_id).await?;
        self.write_block(txn_id, (*latest).into()).await
    }

    pub async fn apply_last(&self, txn: &Txn, subject: &Subject) -> TCResult<()> {
        let latest = *self.latest.read(txn.id()).await?;
        let block = self.read_block(*txn.id(), latest.into()).await?;
        let last_block = if latest > 0 && block.mutations().is_empty() {
            self.read_block(*txn.id(), (latest - 1).into()).await?
        } else {
            block
        };

        if let Some((last_txn_id, ops)) = last_block.mutations().iter().last() {
            for op in ops {
                let result = match op {
                    Mutation::Delete(path, key) => subject.delete(txn, path, key.clone()).await,
                    Mutation::Put(path, key, value) => {
                        self.resolve(txn, value.clone())
                            .and_then(|value| subject.put(txn, path, key.clone(), value))
                            .await
                    }
                };

                if let Err(cause) = result {
                    return Err(TCError::internal(format!(
                        "error replaying last transaction {}: {}",
                        last_txn_id, cause
                    )));
                }
            }
        }

        Ok(())
    }

    pub async fn replicate(&self, txn: &Txn, subject: &Subject, other: Self) -> TCResult<()> {
        debug!("replicate chain history");

        let txn_id = *txn.id();

        let (latest, other_latest) =
            try_join!(self.latest.read(&txn_id), other.latest.read(&txn_id))?;

        if (*latest) > (*other_latest) {
            return Err(TCError::bad_request(
                "cannot replicate from chain with fewer blocks",
                *latest,
            ));
        }

        const ERR_DIVERGENT: &str = "chain to replicate diverges at block";
        for i in 0u64..*latest {
            let block = self.read_block(txn_id, i.into()).await?;
            let other = other.read_block(txn_id, i.into()).await?;
            if &*block != &*other {
                return Err(TCError::bad_request(ERR_DIVERGENT, i));
            }
        }

        let mut i = *latest;
        loop {
            debug!("copy history from block {}", i);
            let source = other.read_block(txn_id, i).await?;
            let mut dest = self.write_block(txn_id, i).await?;

            for (past_txn_id, ops) in source.mutations() {
                let append = !dest.mutations().contains_key(past_txn_id);

                for op in ops.iter().cloned() {
                    debug!("replicating mutation at {}: {}", past_txn_id, op);

                    let result = match op {
                        Mutation::Delete(path, key) => {
                            if append {
                                dest.append_delete(*past_txn_id, path.clone(), key.clone());
                            }

                            subject.delete(txn, &path, key).await
                        }
                        Mutation::Put(path, key, value) => {
                            let value = other.resolve(txn, value).await?;
                            let value_ref = self.save_state(txn.clone(), value.clone()).await?;

                            if append {
                                dest.append_put(*past_txn_id, path.clone(), key.clone(), value_ref);
                            }

                            subject.put(txn, &path, key, value).await
                        }
                    };

                    if let Err(cause) = result {
                        return Err(TCError::bad_request(
                            format!("error at {} while replicating chain", past_txn_id),
                            cause,
                        ));
                    }
                }
            }

            let (source_hash, dest_hash) = try_join!(source.hash(), dest.hash())?;
            if source_hash != dest_hash {
                debug!("source {:?}", &*source);
                debug!("dest {:?}", &*dest);

                return Err(TCError::bad_request(
                    "error replicating chain",
                    format!("hashes diverge at block {}", i),
                ));
            }

            i += 1;

            if other.contains_block(txn.id(), i).await? {
                self.create_next_block(*txn.id()).await?;
            } else {
                break;
            }
        }

        Ok(())
    }

    pub async fn resolve(&self, txn: &Txn, scalar: Scalar) -> TCResult<State> {
        debug!("History::resolve {}", scalar);

        type OpSubject = crate::scalar::Subject;

        if let Scalar::Ref(tc_ref) = scalar {
            if let TCRef::Op(OpRef::Get((OpSubject::Ref(hash, classpath), schema))) = *tc_ref {
                let class = CollectionType::from_path(&classpath).ok_or_else(|| {
                    TCError::internal(format!("invalid Collection type: {}", classpath))
                })?;

                self.resolve_inner(txn, hash.into(), schema, class)
                    .map_ok(State::from)
                    .await
            } else {
                error!("invalid subject for historical Chain state {}", tc_ref);

                Err(TCError::internal(format!(
                    "invalid subject for historical Chain state {}",
                    tc_ref
                )))
            }
        } else {
            Ok(scalar.into())
        }
    }

    async fn resolve_inner(
        &self,
        txn: &Txn,
        hash: Id,
        schema: Scalar,
        class: CollectionType,
    ) -> TCResult<Collection> {
        debug!("resolve historical collection value of type {}", class);

        match class {
            CollectionType::BTree(_) => {
                fn schema_err<I: fmt::Display>(info: I) -> TCError {
                    TCError::internal(format!(
                        "invalid BTree schema for historical Chain state: {}",
                        info
                    ))
                }

                let schema = Value::try_cast_from(schema, |v| schema_err(v))?;
                let schema = schema.try_cast_into(|v| schema_err(v))?;

                let file = self.dir.get_file(txn.id(), &hash).await?.ok_or_else(|| {
                    TCError::internal(format!("Chain is missing historical state {}", hash))
                })?;

                let btree = BTreeFile::load(txn, schema, file).await?;
                Ok(Collection::BTree(btree.into()))
            }
            CollectionType::Table(_) => {
                fn schema_err<I: fmt::Display>(info: I) -> TCError {
                    TCError::internal(format!(
                        "invalid Table schema for historical Chain state: {}",
                        info
                    ))
                }

                let schema = Value::try_cast_from(schema, |v| schema_err(v))?;
                let schema = schema.try_cast_into(|v| schema_err(v))?;

                let dir = self.dir.get_dir(txn.id(), &hash).await?;
                let dir = dir.ok_or_else(|| {
                    TCError::internal(format!("missing historical Chain state {}", hash))
                })?;

                debug!("dir contents {:?}", dir.entry_ids(txn.id()).await?);
                let table = TableIndex::load(txn, schema, dir).await?;
                Ok(Collection::Table(table.into()))
            }

            #[cfg(feature = "tensor")]
            CollectionType::Tensor(tt) => {
                let schema = cast_into_tensor_schema(schema)?;

                match tt {
                    TensorType::Dense => {
                        let file = self.dir.get_file(txn.id(), &hash).await?;
                        let file = file.ok_or_else(|| {
                            TCError::internal(format!("missing historical Chain state {}", hash))
                        })?;

                        let tensor = DenseTensor::load(txn, schema, file).await?;
                        Ok(Collection::Tensor(tensor.into()))
                    }
                    TensorType::Sparse => {
                        let dir = self.dir.get_dir(txn.id(), &hash).await?;
                        let dir = dir.ok_or_else(|| {
                            TCError::internal(format!("missing historical Chain state {}", hash))
                        })?;

                        debug!("dir contents {:?}", dir.entry_ids(txn.id()).await?);
                        let tensor = SparseTensor::load(txn, schema, dir).await?;
                        Ok(Collection::Tensor(tensor.into()))
                    }
                }
            }
        }
    }
}

const SCHEMA: () = ();

#[async_trait]
impl Persist<fs::Dir> for History {
    type Schema = ();
    type Store = fs::Dir;
    type Txn = Txn;

    fn schema(&self) -> &() {
        &SCHEMA
    }

    async fn load(txn: &Txn, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        let txn_id = txn.id();

        let file: fs::File<ChainBlock> = dir
            .get_file(txn_id, &CHAIN.into())
            .await?
            .ok_or_else(|| TCError::internal("Chain has no history file"))?;

        let dir = dir
            .get_dir(txn_id, &DATA.into())
            .await?
            .ok_or_else(|| TCError::internal("Chain has no data directory"))?;

        let mut last_hash = Bytes::from(NULL_HASH);
        let mut latest = 0;

        loop {
            let block = file.read_block(*txn_id, latest.into()).await?;
            if block.last_hash() == &last_hash {
                last_hash = block.last_hash().clone();
            } else {
                return Err(TCError::internal(format!(
                    "block {} hash does not match previous block",
                    latest
                )));
            }

            if file.contains_block(txn_id, &(latest + 1).into()).await? {
                latest += 1;
            } else {
                break;
            }
        }

        Ok(History::new(latest, dir, file))
    }
}

#[async_trait]
impl Transact for History {
    async fn commit(&self, txn_id: &TxnId) {
        join!(self.file.commit(txn_id), self.dir.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.file.finalize(txn_id), self.dir.finalize(txn_id));
    }
}

#[async_trait]
impl de::FromStream for History {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(HistoryVisitor { txn }).await
    }
}

struct HistoryVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for HistoryVisitor {
    type Value = History;

    fn expecting() -> &'static str {
        "Chain history"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let txn_id = *self.txn.id();
        let dir = self.txn.context().clone();

        let file: fs::File<ChainBlock> = dir
            .create_file(txn_id, CHAIN.into(), ChainType::default())
            .map_err(de::Error::custom)
            .await?;

        let dir = dir
            .create_dir(txn_id, DATA.into())
            .map_err(de::Error::custom)
            .await?;

        let first_block = ChainBlock::new(NULL_HASH);
        file.create_block(txn_id, 0u64.into(), first_block)
            .map_err(de::Error::custom)
            .await?;

        let history = History::new(0, dir, file);

        let subcontext = |i: u64| self.txn.subcontext(i.into()).map_err(de::Error::custom);

        let mut i = 0u64;
        let txn = subcontext(i).await?;

        if let Some(state) = seq.next_element::<State>(txn.clone()).await? {
            let (hash, block_data): (Bytes, Map<Tuple<State>>) = state
                .try_cast_into(|s| TCError::bad_request("invalid Chain block", s))
                .map_err(de::Error::custom)?;

            if hash != NULL_HASH {
                let hash = hex::encode(hash);
                let null_hash = hex::encode(NULL_HASH);
                return Err(de::Error::invalid_value(
                    format!("initial block hash {}", hash),
                    format!("null hash {}", null_hash),
                ));
            }

            let mutations = parse_block_state(&history, txn.clone(), block_data)
                .map_err(de::Error::custom)
                .await?;

            let mut block = history
                .write_block(txn_id, i)
                .map_err(de::Error::custom)
                .await?;

            *block = ChainBlock::with_mutations(hash, mutations);
        }

        i += 1;
        while let Some(state) = seq.next_element::<State>(subcontext(i).await?).await? {
            let (hash, block_data): (Bytes, Map<Tuple<State>>) = state
                .try_cast_into(|s| TCError::bad_request("invalid Chain block", s))
                .map_err(de::Error::custom)?;

            let mut block = history
                .create_next_block(txn_id)
                .map_err(de::Error::custom)
                .await?
                .write()
                .await;

            if block.last_hash() != &hash {
                let hash = hex::encode(hash);
                let last_hash = hex::encode(block.last_hash());
                return Err(de::Error::invalid_value(
                    format!("block with last hash {}", hash),
                    format!("block with last hash {}", last_hash),
                ));
            }

            let mutations = parse_block_state(&history, txn.clone(), block_data)
                .map_err(de::Error::custom)
                .await?;

            *block = ChainBlock::with_mutations(hash, mutations);

            i += 1;
        }

        Ok(history)
    }
}

async fn parse_block_state(
    history: &History,
    txn: Txn,
    block_data: Map<Tuple<State>>,
) -> TCResult<BTreeMap<TxnId, Vec<Mutation>>> {
    let mut mutations = BTreeMap::new();

    for (past_txn_id, ops) in block_data.into_iter() {
        let past_txn_id = past_txn_id.to_string().parse()?;

        let mut parsed = Vec::with_capacity(ops.len());

        for op in ops.into_iter() {
            if op.matches::<(TCPathBuf, Value)>() {
                let (path, key) = op.opt_cast_into().unwrap();
                parsed.push(Mutation::Delete(path, key));
            } else if op.matches::<(TCPathBuf, Value, State)>() {
                let (path, key, value) = op.opt_cast_into().unwrap();
                let value = history.save_state(txn.clone(), value).await?;
                parsed.push(Mutation::Put(path, key, value));
            } else {
                return Err(TCError::bad_request(
                    "unable to parse historical mutation",
                    op,
                ));
            }
        }

        mutations.insert(past_txn_id, parsed);
    }

    Ok(mutations)
}

pub type HistoryView<'en> =
    en::SeqStream<TCError, HistoryBlockView<'en>, TCTryStream<'en, HistoryBlockView<'en>>>;

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for History {
    type Txn = Txn;
    type View = HistoryView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        debug!("History::into_view");

        let txn_id = *txn.id();
        let latest = self.latest.read(&txn_id).await?;

        let file = self.file.clone();
        let read_block = move |block_id| Box::pin(file.clone().read_block_owned(txn_id, block_id));

        let seq = stream::iter(0..((*latest) + 1))
            .map(BlockId::from)
            .inspect(|id| debug!("encoding history block {}", id))
            .then(read_block)
            .map_ok(move |block| {
                let this = self.clone();
                let txn = txn.clone();
                let map =
                    stream::iter(block.mutations().clone()).map(move |(past_txn_id, mutations)| {
                        debug!("reading block mutations");

                        let this = this.clone();
                        let txn = txn.clone();
                        let mutations = stream::iter(mutations)
                            .then(move |op| Box::pin(load_history(this.clone(), op, txn.clone())));

                        let mutations: TCTryStream<'en, MutationView<'en>> = Box::pin(mutations);
                        let mutations = en::SeqStream::from(mutations);
                        Ok((past_txn_id, mutations))
                    });

                let map: TCTryStream<'en, (TxnId, MutationViewSeq<'en>)> = Box::pin(map);
                (block.last_hash().clone(), en::MapStream::from(map))
            });

        let seq: TCTryStream<'en, HistoryBlockView<'en>> = Box::pin(seq);
        Ok(en::SeqStream::from(seq))
    }
}

async fn load_history<'a>(history: History, op: Mutation, txn: Txn) -> TCResult<MutationView<'a>> {
    match op {
        Mutation::Delete(path, key) => Ok(MutationView::Delete(path, key)),
        Mutation::Put(path, key, value) if value.is_ref() => {
            debug!("historical mutation: PUT {}: {} <- {}", path, key, value);

            let value = history
                .resolve(&txn, value)
                .map_err(|err| {
                    error!("unable to load historical Chain data: {}", err);
                    err
                })
                .await?;

            let value = value.into_view(txn).await?;
            Ok(MutationView::Put(path, key, value))
        }
        Mutation::Put(path, key, value) => {
            let value = State::from(value).into_view(txn.clone()).await?;

            Ok(MutationView::Put(path, key, value))
        }
    }
}

type MutationViewSeq<'en> =
    en::SeqStream<TCError, MutationView<'en>, TCTryStream<'en, MutationView<'en>>>;

type HistoryBlockView<'en> = (
    Bytes,
    en::MapStream<
        TCError,
        TxnId,
        MutationViewSeq<'en>,
        TCTryStream<'en, (TxnId, MutationViewSeq<'en>)>,
    >,
);

pub enum MutationView<'en> {
    Delete(TCPathBuf, Value),
    Put(TCPathBuf, Value, StateView<'en>),
}

impl<'en> en::IntoStream<'en> for MutationView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Delete(path, key) => (path, key).into_stream(encoder),
            Self::Put(path, key, value) => (path, key, value).into_stream(encoder),
        }
    }
}

#[cfg(feature = "tensor")]
fn cast_tensor_schema(tensor: &Tensor) -> Value {
    let shape: Tuple<Value> = tensor
        .shape()
        .to_vec()
        .into_iter()
        .map(Value::from)
        .collect();

    let dtype = tc_value::ValueType::from(tensor.dtype()).path();
    Value::Tuple(Tuple::from(vec![shape.into(), dtype.into()]))
}

#[cfg(feature = "tensor")]
fn cast_into_tensor_schema(scalar: Scalar) -> TCResult<tc_tensor::Schema> {
    use std::convert::TryInto;
    use tc_value::ValueType;

    let (shape, dtype): (Vec<u64>, TCPathBuf) = match scalar {
        Scalar::Value(Value::Tuple(schema)) => {
            schema.try_cast_into(|v| TCError::internal(format!("invalid Tensor schema: {}", v)))
        }
        Scalar::Tuple(schema) => {
            schema.try_cast_into(|v| TCError::internal(format!("invalid Tensor schema: {}", v)))
        }
        other => Err(TCError::internal(format!(
            "invalid Tensor schema: {}",
            other
        ))),
    }?;

    let shape = tc_tensor::Shape::from(shape);

    let dtype = ValueType::from_path(&dtype)
        .ok_or_else(|| TCError::internal(format!("invalid data type for Tensor: {}", dtype)))?;
    let dtype = dtype.try_into().map_err(TCError::internal)?;

    Ok((shape, dtype))
}
