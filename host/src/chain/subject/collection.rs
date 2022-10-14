//! A [`Collection`] which is the `Subject` of a `Chain`

use std::fmt;

use async_hash::hash_try_stream;
use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use log::{debug, trace};
use sha2::digest::Output;
use sha2::Sha256;

use tc_btree::{BTreeInstance, BTreeType};
use tc_error::*;
use tc_table::TableStream;
#[cfg(feature = "tensor")]
use tc_tensor::TensorPersist;
use tc_transact::fs::{Dir, DirRead, DirWrite, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::*;

use crate::collection::{BTree, BTreeFile, Collection, Table, TableIndex};
#[cfg(feature = "tensor")]
use crate::collection::{
    DenseAccess, DenseTensor, DenseTensorFile, SparseAccess, SparseTable, SparseTensor, Tensor,
    TensorType,
};
use crate::fs;
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::CollectionSchema;

/// A [`Collection`] which is the `Subject` of a `Chain`
#[derive(Clone)]
pub enum SubjectCollection {
    BTree(BTreeFile),
    Table(TableIndex),
    #[cfg(feature = "tensor")]
    Dense(DenseTensor<DenseTensorFile>),
    #[cfg(feature = "tensor")]
    Sparse(SparseTensor<SparseTable>),
}

impl SubjectCollection {
    pub(super) fn from_collection(collection: Collection) -> TCResult<Self> {
        match collection {
            Collection::BTree(BTree::File(btree)) => Ok(SubjectCollection::BTree(btree)),
            Collection::Table(Table::Table(table)) => Ok(SubjectCollection::Table(table)),

            #[cfg(feature = "tensor")]
            Collection::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => dense
                    .as_persistent()
                    .map(SubjectCollection::Dense)
                    .ok_or_else(|| {
                        TCError::unsupported("Chain expected a Dense tensor, not a view")
                    }),

                Tensor::Sparse(sparse) => sparse
                    .as_persistent()
                    .map(SubjectCollection::Sparse)
                    .ok_or_else(|| {
                        TCError::unsupported("Chain expected a Sparse tensor, not a view")
                    }),
            },
            other => Err(TCError::bad_request(
                "Chain expected a Collection, not",
                other,
            )),
        }
    }

    /// Create a new [`SubjectCollection`] with the given `Schema`.
    pub fn create(
        schema: CollectionSchema,
        dir: &fs::Dir,
        txn_id: TxnId,
        name: Id,
    ) -> TCBoxTryFuture<Self> {
        Box::pin(async move {
            debug!("SubjectCollection::create {}", schema);

            let mut dir = dir.write(txn_id).await?;
            trace!("SubjectCollection::create got dir write lock");

            match schema {
                #[cfg(feature = "tensor")]
                CollectionSchema::Dense(schema) => {
                    let file = dir.create_file(name, TensorType::Dense)?;
                    DenseTensor::create(file, schema, txn_id)
                        .map_ok(Self::Dense)
                        .await
                }
                #[cfg(feature = "tensor")]
                CollectionSchema::Sparse(schema) => {
                    let dir = dir.create_dir(name)?;
                    SparseTensor::create(&dir, schema, txn_id)
                        .map_ok(Self::Sparse)
                        .await
                }
                CollectionSchema::Table(schema) => {
                    let dir = dir.create_dir(name)?;
                    TableIndex::create(&dir, schema, txn_id)
                        .map_ok(Self::Table)
                        .await
                }
                CollectionSchema::BTree(schema) => {
                    let file = dir.create_file(name, BTreeType::default())?;
                    BTreeFile::create(file, schema, txn_id)
                        .map_ok(Self::BTree)
                        .await
                }
            }
        })
    }

    pub(super) fn load<'a>(
        txn: &'a Txn,
        schema: CollectionSchema,
        dir: &'a fs::Dir,
        name: Id,
    ) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            debug!("SubjectCollection::load");

            let txn_id = *txn.id();
            let container = dir.read(txn_id).await?;

            trace!("SubjectCollection::load got read lock on container dir");

            match schema {
                CollectionSchema::BTree(schema) => {
                    if let Some(file) = container.get_file(&name)? {
                        trace!("SubjectCollection::load loading BTree from existing file");
                        BTreeFile::load(txn, schema, file).map_ok(Self::BTree).await
                    } else {
                        trace!("SubjectCollection::load creating new BTree");
                        std::mem::drop(container);
                        Self::create(CollectionSchema::BTree(schema), dir, txn_id, name).await
                    }
                }

                CollectionSchema::Table(schema) => {
                    if let Some(dir) = container.get_dir(&name)? {
                        TableIndex::load(txn, schema, dir.clone())
                            .map_ok(Self::Table)
                            .await
                    } else {
                        std::mem::drop(container);
                        Self::create(CollectionSchema::Table(schema), dir, txn_id, name).await
                    }
                }

                #[cfg(feature = "tensor")]
                CollectionSchema::Dense(schema) => {
                    if let Some(file) = container.get_file(&name)? {
                        DenseTensor::load(txn, schema, file)
                            .map_ok(Self::Dense)
                            .await
                    } else {
                        std::mem::drop(container);
                        Self::create(CollectionSchema::Dense(schema), dir, txn_id, name).await
                    }
                }

                #[cfg(feature = "tensor")]
                CollectionSchema::Sparse(schema) => {
                    if let Some(dir) = container.get_dir(&name)? {
                        SparseTensor::load(txn, schema, dir)
                            .map_ok(Self::Sparse)
                            .await
                    } else {
                        std::mem::drop(container);
                        Self::create(CollectionSchema::Sparse(schema), dir, txn_id, name).await
                    }
                }
            }
        })
    }

    pub(super) fn restore<'a>(&'a self, txn: &'a Txn, backup: Collection) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let txn_id = *txn.id();

            match self {
                Self::BTree(btree) => match backup {
                    Collection::BTree(BTree::File(backup)) => btree.restore(&backup, txn_id).await,
                    other => Err(TCError::bad_request("cannot restore a BTree from", other)),
                },
                Self::Table(table) => match backup {
                    Collection::Table(Table::Table(backup)) => table.restore(&backup, txn_id).await,
                    other => Err(TCError::bad_request("cannot restore a Table from", other)),
                },

                #[cfg(feature = "tensor")]
                Self::Dense(tensor) => match backup {
                    Collection::Tensor(Tensor::Dense(backup)) => {
                        let file = txn
                            .context()
                            .create_file_unique(txn_id, TensorType::Dense)
                            .await?;

                        let backup =
                            tc_transact::fs::CopyFrom::copy_from(backup, file, txn).await?;

                        tensor.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request(
                        "cannot restore a dense Tensor from",
                        other,
                    )),
                },

                #[cfg(feature = "tensor")]
                Self::Sparse(tensor) => match backup {
                    Collection::Tensor(Tensor::Sparse(backup)) => {
                        let dir = txn.context().create_dir_unique(txn_id).await?;
                        let backup = tc_transact::fs::CopyFrom::copy_from(backup, dir, txn).await?;
                        tensor.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request(
                        "cannot restore a sparse Tensor from",
                        other,
                    )),
                },
            }
        })
    }

    pub fn schema(&self) -> CollectionSchema {
        match self {
            Self::BTree(btree) => CollectionSchema::BTree(BTreeInstance::schema(btree).clone()),
            Self::Table(table) => CollectionSchema::Table(table.schema().clone()),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => CollectionSchema::Dense(dense.schema().clone()),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => CollectionSchema::Sparse(sparse.schema().clone()),
        }
    }

    pub async fn into_state(self, _txn_id: TxnId) -> TCResult<State> {
        let collection = match self {
            Self::BTree(btree) => btree.into(),
            Self::Table(table) => table.into(),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => dense.into(),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.into(),
        };

        Ok(State::Collection(collection))
    }

    pub fn hash<'a>(self, txn: Txn) -> TCBoxTryFuture<'a, Output<Sha256>> {
        Box::pin(async move {
            // TODO: should this be consolidated with Collection::hash?
            match self {
                Self::BTree(btree) => {
                    let keys = btree.keys(*txn.id()).await?;
                    hash_try_stream::<Sha256, _, _, _>(keys).await
                }
                Self::Table(table) => {
                    let rows = table.rows(*txn.id()).await?;
                    hash_try_stream::<Sha256, _, _, _>(rows).await
                }
                #[cfg(feature = "tensor")]
                Self::Dense(dense) => {
                    let elements = dense.into_inner().value_stream(txn).await?;
                    hash_try_stream::<Sha256, _, _, _>(elements).await
                }
                #[cfg(feature = "tensor")]
                Self::Sparse(sparse) => {
                    let filled = sparse.into_inner().filled(txn).await?;
                    hash_try_stream::<Sha256, _, _, _>(filled).await
                }
            }
        })
    }
}

#[async_trait]
impl Transact for SubjectCollection {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain subject collection");

        match self {
            Self::BTree(btree) => {
                btree.commit(txn_id).await;
            }
            Self::Table(table) => {
                table.commit(txn_id).await;
            }
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => {
                tensor.commit(txn_id).await;
            }
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => {
                tensor.commit(txn_id).await;
            }
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize chain subject collection");

        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => tensor.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => tensor.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl de::FromStream for SubjectCollection {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let collection = Collection::from_stream(txn, decoder).await?;
        Self::from_collection(collection).map_err(de::Error::custom)
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for SubjectCollection {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::BTree(btree) => State::from(BTree::File(btree)).into_view(txn).await,
            Self::Table(table) => State::from(Table::Table(table)).into_view(txn).await,
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => State::from(Tensor::from(tensor)).into_view(txn).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => State::from(Tensor::from(tensor)).into_view(txn).await,
        }
    }
}

impl From<SubjectCollection> for Collection {
    fn from(subject: SubjectCollection) -> Self {
        match subject {
            SubjectCollection::BTree(btree) => Collection::BTree(btree.into()),
            SubjectCollection::Table(table) => Collection::Table(table.into()),
            #[cfg(feature = "tensor")]
            SubjectCollection::Dense(dense) => Collection::Tensor(dense.into()),
            #[cfg(feature = "tensor")]
            SubjectCollection::Sparse(sparse) => Collection::Tensor(sparse.into()),
        }
    }
}

impl fmt::Display for SubjectCollection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => write!(f, "chain Subject, {}", btree.class()),
            Self::Table(table) => write!(f, "chain Subject, {}", table.class()),
            #[cfg(feature = "tensor")]
            Self::Dense(_) => write!(f, "chain Subject, {}", TensorType::Dense),
            #[cfg(feature = "tensor")]
            Self::Sparse(_) => write!(f, "chain Subject, {}", TensorType::Sparse),
        }
    }
}
