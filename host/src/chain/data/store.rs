use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use futures::future::TryFutureExt;
use log::{debug, error};
use safecast::*;

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableInstance;
#[cfg(feature = "tensor")]
use tc_tensor::TensorAccess;
use tc_transact::fs::*;
use tc_transact::{Transact, Transaction};
use tc_value::Value;
use tcgeneric::{Id, Instance, NativeClass};

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

        if state.is_ref() {
            return Err(TCError::bad_request(
                "cannot update Chain with reference: {}",
                state,
            ));
        }

        let hash = state
            .clone()
            .hash(txn.clone())
            .map_ok(Id::from_hash)
            .await?;

        debug!("computed hash of {}: {}", state, hash);

        let txn_id = *txn.id();
        let mut dir = self.dir.write(txn_id).await?;

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
                        let file = dir.create_file(hash.clone())?;

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
                    let schema = table.schema().clone();
                    let classpath = TableType::default().path();

                    if dir.contains(&hash) {
                        debug!("Table with hash {} is already saved", hash);
                    } else {
                        let dir = dir.create_dir(hash.clone())?;
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
                    debug!("chain data store copying {} into data store", tensor);

                    let shape = tensor.shape().clone();
                    let dtype = tensor.dtype();
                    let schema = tc_tensor::Schema { shape, dtype };
                    let classpath = tensor.class().path();

                    if dir.contains(&hash) {
                        debug!("Tensor with hash {} is already saved", tensor);
                    } else {
                        match tensor {
                            Tensor::Dense(dense) => {
                                debug!(
                                    "chain data store creating destination file for {}...",
                                    dense
                                );

                                let file = dir.create_file(hash.clone())?;

                                debug!("chain data store created destination file for {}", dense);
                                DenseTensor::copy_from(dense, file, txn).await?;
                                debug!("saved Tensor with hash {}", hash);
                            }
                            Tensor::Sparse(sparse) => {
                                let dir = dir.create_dir(hash.clone())?;
                                SparseTensor::copy_from(sparse, dir, txn).await?;
                                debug!("saved Tensor with hash {}", hash);
                            }
                        };
                    }

                    let schema: Value = schema.cast_into();
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

        let dir = self.dir.read(*txn.id()).await?;

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

                let file = dir.get_file(&hash)?.ok_or_else(|| {
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

                let dir = dir.get_dir(&hash)?;
                let dir = dir.ok_or_else(|| {
                    TCError::internal(format!("missing historical Chain state {}", hash))
                })?;

                let table = TableIndex::load(txn, schema, dir).await?;
                Ok(Collection::Table(table.into()))
            }

            #[cfg(feature = "tensor")]
            CollectionType::Tensor(tt) => {
                let schema: Value = schema.try_cast_into(|s| {
                    TCError::internal(format!("invalid Tensor schema: {}", s))
                })?;
                let schema = schema.try_cast_into(|v| {
                    TCError::internal(format!("invalid Tensor schema: {}", v))
                })?;

                match tt {
                    TensorType::Dense => {
                        let file = dir.get_file(&hash)?;
                        let file = file.ok_or_else(|| {
                            TCError::internal(format!("missing historical Chain state {}", hash))
                        })?;

                        let tensor = DenseTensor::load(txn, schema, file).await?;
                        Ok(Collection::Tensor(tensor.into()))
                    }
                    TensorType::Sparse => {
                        let dir = dir.get_dir(&hash)?;
                        let dir = dir.ok_or_else(|| {
                            TCError::internal(format!("missing historical Chain state {}", hash))
                        })?;

                        let tensor = SparseTensor::load(txn, schema, dir).await?;
                        Ok(Collection::Tensor(tensor.into()))
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Transact for Store {
    type Commit = <fs::Dir as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.dir.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}
