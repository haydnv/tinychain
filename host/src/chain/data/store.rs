use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use futures::future::TryFutureExt;
use log::debug;
use safecast::*;

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableInstance;
#[cfg(feature = "tensor")]
use tc_tensor::TensorAccess;
use tc_transact::fs::*;
use tc_transact::{AsyncHash, Transact, Transaction};
use tc_value::Value;
use tcgeneric::{Id, NativeClass};

use crate::collection::{BTreeFile, BTreeType, Collection, CollectionType, TableIndex, TableType};
#[cfg(feature = "tensor")]
use crate::collection::{DenseTensor, SparseTensor, Tensor, TensorType};
use crate::fs;
use crate::scalar::{OpRef, Scalar, TCRef};
use crate::state::State;
use crate::transact::TxnId;
use crate::txn::Txn;

#[derive(Clone)]
pub struct Store {
    dir: fs::Dir,
}

impl Store {
    pub fn new(dir: fs::Dir) -> Self {
        Self { dir }
    }

    pub async fn save_state(&self, txn: &Txn, state: State) -> TCResult<Scalar> {
        debug!("chain data store saving state {}...", state);

        let hash = state.clone().hash(txn).map_ok(Id::from_hash).await?;

        debug!("computed hash of {}: {}", state, hash);

        let txn_id = *txn.id();
        let dir = self.dir.write(txn_id).await?;

        // TODO: it should be possible to lock the directory listing,
        // to combine the calls to `self.dir.contains` with `self.dir.create...`
        match state {
            State::Collection(collection) => match collection {
                Collection::BTree(btree) => {
                    let schema = btree.schema().to_vec();
                    let classpath = BTreeType::default().path();

                    if dir.contains(&hash) {
                        debug!("BTree with hash {} is already saved", hash);
                    } else {
                        let store = dir.create_store(hash.clone());
                        BTreeFile::copy_from(txn, store, btree).await?;
                        debug!("saved BTree with hash {}", hash);
                    }

                    Ok(OpRef::Get((
                        (hash.into(), classpath).into(),
                        Value::from_iter(schema).into(),
                    ))
                    .into())
                }

                Collection::Table(table) => {
                    let schema = table.schema().clone();
                    let classpath = TableType::default().path();

                    if dir.contains(&hash) {
                        debug!("Table with hash {} is already saved", hash);
                    } else {
                        let store = dir.create_store(hash.clone());
                        TableIndex::copy_from(txn, store, table).await?;
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
                    debug!("chain data store copying {} into data store", tensor);

                    let shape = tensor.shape().clone();
                    let dtype = tensor.dtype();
                    let schema = tc_tensor::Schema { shape, dtype };
                    let classpath = tcgeneric::Instance::class(&tensor).path();

                    if dir.contains(&hash) {
                        debug!("Tensor with hash {} is already saved", tensor);
                    } else {
                        match tensor {
                            Tensor::Dense(dense) => {
                                debug!(
                                    "chain data store creating destination file for {}...",
                                    dense
                                );

                                let store = dir.create_store(hash.clone());

                                debug!("chain data store created destination file for {}", dense);
                                DenseTensor::copy_from(txn, store, dense).await?;
                                debug!("saved Tensor with hash {}", hash);
                            }
                            Tensor::Sparse(sparse) => {
                                let store = dir.create_store(hash.clone());
                                SparseTensor::copy_from(txn, store, sparse).await?;
                                debug!("saved Tensor with hash {}", hash);
                            }
                        };
                    }

                    let schema: Value = schema.cast_into();
                    Ok(OpRef::Get(((hash.into(), classpath).into(), schema.into())).into())
                }
            },
            State::Scalar(value) => Ok(value),
            other => other.try_cast_into(|s| bad_request!("Chain does not support value {}", s)),
        }
    }

    pub async fn resolve(&self, txn_id: TxnId, scalar: Scalar) -> TCResult<State> {
        debug!("History::resolve {}", scalar);

        type OpSubject = crate::scalar::Subject;

        if let Scalar::Ref(tc_ref) = scalar {
            if let TCRef::Op(OpRef::Get((OpSubject::Ref(hash, classpath), schema))) = *tc_ref {
                let class = CollectionType::from_path(&classpath)
                    .ok_or_else(|| unexpected!("invalid Collection type: {}", classpath))?;

                let dir = self.dir.read(txn_id).await?;
                Self::resolve_inner(dir, txn_id, hash.into(), schema, class).map(State::from)
            } else {
                Err(unexpected!(
                    "invalid subject for historical Chain state {}",
                    tc_ref
                ))
            }
        } else {
            Ok(scalar.into())
        }
    }

    fn resolve_inner(
        dir: fs::DirReadGuard,
        txn_id: TxnId,
        hash: Id,
        schema: Scalar,
        class: CollectionType,
    ) -> TCResult<Collection> {
        debug!("resolve historical collection value of type {}", class);

        match class {
            CollectionType::BTree(_) => {
                fn schema_err<I: fmt::Display>(info: I) -> TCError {
                    unexpected!("invalid BTree schema for historical Chain state: {}", info)
                }

                let schema = Value::try_cast_from(schema, |v| schema_err(v))?;
                let schema = schema.try_cast_into(|v| schema_err(v))?;

                let store = dir
                    .get_store(hash)
                    .ok_or_else(|| unexpected!("missing historical state"))?;

                BTreeFile::load(txn_id, schema, store).map(|btree| Collection::BTree(btree.into()))
            }

            CollectionType::Table(_) => {
                fn schema_err<I: fmt::Display>(info: I) -> TCError {
                    unexpected!("invalid Table schema for historical Chain state: {}", info)
                }

                let schema = Value::try_cast_from(schema, |v| schema_err(v))?;
                let schema = schema.try_cast_into(|v| schema_err(v))?;

                let store = dir
                    .get_store(hash)
                    .ok_or_else(|| unexpected!("missing historical state"))?;

                TableIndex::load(txn_id, schema, store).map(|table| Collection::Table(table.into()))
            }

            #[cfg(feature = "tensor")]
            CollectionType::Tensor(tt) => {
                let schema: Value =
                    schema.try_cast_into(|s| unexpected!("invalid Tensor schema: {}", s))?;

                let schema =
                    schema.try_cast_into(|v| unexpected!("invalid Tensor schema: {}", v))?;

                match tt {
                    TensorType::Dense => {
                        let store = dir
                            .get_store(hash)
                            .ok_or_else(|| unexpected!("missing historical state"))?;

                        DenseTensor::load(txn_id, schema, store)
                            .map(|tensor| Collection::Tensor(tensor.into()))
                    }
                    TensorType::Sparse => {
                        let store = dir
                            .get_store(hash)
                            .ok_or_else(|| unexpected!("missing historical state"))?;

                        SparseTensor::load(txn_id, schema, store)
                            .map(|tensor| Collection::Tensor(tensor.into()))
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Transact for Store {
    type Commit = <fs::Dir as Transact>::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit chain data store at {}", txn_id);

        self.dir.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.dir.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}
