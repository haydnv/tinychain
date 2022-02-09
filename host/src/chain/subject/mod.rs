//! The [`Subject`] of a [`Chain`]

use std::fmt;

use async_hash::{hash_try_stream, Hash};
use async_trait::async_trait;
use destream::de;
use futures::future::{join_all, try_join_all, TryFutureExt};
use log::debug;
use sha2::digest::{Digest, Output};
use sha2::Sha256;

use tc_btree::{BTreeInstance, BTreeType};
use tc_error::*;
use tc_table::TableStream;
#[cfg(feature = "tensor")]
use tc_tensor::TensorPersist;
use tc_transact::fs::{Dir, File, Persist, Restore, Store};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::*;

use crate::collection::{BTree, BTreeFile, Collection, CollectionType, Table, TableIndex};
#[cfg(feature = "tensor")]
use crate::collection::{
    DenseAccess, DenseTensor, DenseTensorFile, SparseAccess, SparseTable, SparseTensor, Tensor,
    TensorType,
};
use crate::fs;
use crate::scalar::{Scalar, ScalarType};
use crate::state::{State, StateType, StateView};
use crate::txn::Txn;

use super::{CollectionSchema, Schema};

pub use map::SubjectMap;

mod map;

const DYNAMIC: Label = label("dynamic");
const SUBJECT: Label = label("subject");

/// A [`Collection`] which is the [`Subject`] of a [`Chain`]
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
    fn from_collection<E: de::Error>(collection: Collection) -> Result<Self, E> {
        const ERR_INVALID: &str =
            "a Chain subject (must be a collection like a BTree, Table, or Tensor";

        match collection {
            Collection::BTree(BTree::File(btree)) => Ok(SubjectCollection::BTree(btree)),
            Collection::Table(Table::Table(table)) => Ok(SubjectCollection::Table(table)),

            #[cfg(feature = "tensor")]
            Collection::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => dense
                    .as_persistent()
                    .map(SubjectCollection::Dense)
                    .ok_or_else(|| de::Error::invalid_type("a Dense tensor view", ERR_INVALID)),

                Tensor::Sparse(sparse) => sparse
                    .as_persistent()
                    .map(SubjectCollection::Sparse)
                    .ok_or_else(|| de::Error::invalid_type("a Sparse tensor view", ERR_INVALID)),
            },
            other => Err(de::Error::invalid_type(other, ERR_INVALID)),
        }
    }

    /// Create a new [`SubjectCollection`] with the given [`Schema`].
    pub fn create<'a>(
        schema: CollectionSchema,
        dir: &'a fs::Dir,
        txn_id: TxnId,
    ) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            match schema {
                #[cfg(feature = "tensor")]
                CollectionSchema::Dense(schema) => {
                    let file = dir
                        .create_file(txn_id, SUBJECT.into(), TensorType::Dense)
                        .await?;

                    DenseTensor::create(file, schema, txn_id)
                        .map_ok(Self::Dense)
                        .await
                }
                #[cfg(feature = "tensor")]
                CollectionSchema::Sparse(schema) => {
                    let dir = dir.create_dir(txn_id, SUBJECT.into()).await?;
                    let tensor = SparseTensor::create(&dir, schema, txn_id)
                        .map_ok(Self::Sparse)
                        .await?;

                    Ok(tensor)
                }
                CollectionSchema::Table(schema) => {
                    TableIndex::create(dir, schema, txn_id)
                        .map_ok(Self::Table)
                        .await
                }
                CollectionSchema::BTree(schema) => {
                    let file = dir
                        .create_file(txn_id, SUBJECT.into(), BTreeType::default())
                        .await?;

                    BTreeFile::create(file, schema, txn_id)
                        .map_ok(Self::BTree)
                        .await
                }
            }
        })
    }

    fn load<'a>(
        txn: &'a Txn,
        schema: CollectionSchema,
        dir: &'a fs::Dir,
    ) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            let txn_id = *txn.id();

            match schema {
                CollectionSchema::BTree(schema) => {
                    if let Some(file) = dir.get_file(txn_id, &SUBJECT.into()).await? {
                        BTreeFile::load(txn, schema, file).map_ok(Self::BTree).await
                    } else {
                        Self::create(CollectionSchema::BTree(schema), dir, txn_id).await
                    }
                }
                CollectionSchema::Table(schema) => {
                    if dir.is_empty(txn_id).await? {
                        Self::create(CollectionSchema::Table(schema), dir, txn_id).await
                    } else {
                        TableIndex::load(txn, schema, dir.clone())
                            .map_ok(Self::Table)
                            .await
                    }
                }

                #[cfg(feature = "tensor")]
                CollectionSchema::Dense(schema) => {
                    if let Some(file) = dir.get_file(txn_id, &SUBJECT.into()).await? {
                        DenseTensor::load(txn, schema, file)
                            .map_ok(Self::Dense)
                            .await
                    } else {
                        Self::create(CollectionSchema::Dense(schema), dir, txn_id).await
                    }
                }
                #[cfg(feature = "tensor")]
                CollectionSchema::Sparse(schema) => {
                    if let Some(dir) = dir.get_dir(txn_id, &SUBJECT.into()).await? {
                        SparseTensor::load(txn, schema, dir)
                            .map_ok(Self::Sparse)
                            .await
                    } else {
                        Self::create(CollectionSchema::Sparse(schema), dir, txn_id).await
                    }
                }
            }
        })
    }

    fn restore<'a>(&'a self, txn: &'a Txn, backup: State) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let txn_id = *txn.id();

            match self {
                Self::BTree(btree) => match backup {
                    State::Collection(Collection::BTree(BTree::File(backup))) => {
                        btree.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request("cannot restore a BTree from", other)),
                },
                Self::Table(table) => match backup {
                    State::Collection(Collection::Table(Table::Table(backup))) => {
                        table.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request("cannot restore a Table from", other)),
                },

                #[cfg(feature = "tensor")]
                Self::Dense(tensor) => match backup {
                    State::Collection(Collection::Tensor(Tensor::Dense(backup))) => {
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
                    State::Collection(Collection::Tensor(Tensor::Sparse(backup))) => {
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

impl Instance for SubjectCollection {
    type Class = StateType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => CollectionType::BTree(btree.class()).into(),
            Self::Table(table) => CollectionType::Table(table.class()).into(),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => CollectionType::Tensor(dense.class()).into(),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => CollectionType::Tensor(sparse.class()).into(),
        }
    }
}

#[async_trait]
impl Transact for SubjectCollection {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain subject collection");

        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => tensor.commit(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => tensor.commit(txn_id).await,
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
        Self::from_collection(collection)
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

/// The state whose transactional integrity is protected by a [`Chain`]
#[derive(Clone)]
pub enum Subject {
    Collection(SubjectCollection),
    Map(Map<Subject>),
    Tuple(Tuple<Subject>),
}

impl Subject {
    fn from_state<E: de::Error>(state: State) -> Result<Subject, E> {
        const ERR_INVALID: &str =
            "a Chain subject (must be a collection like a BTree, Table, or Tensor";

        match state {
            State::Collection(collection) => {
                SubjectCollection::from_collection(collection).map(Self::Collection)
            }

            State::Map(map) => {
                let subject = map
                    .into_iter()
                    .map(|(name, state)| Subject::from_state(state).map(|subject| (name, subject)))
                    .collect::<Result<Map<Subject>, E>>()?;

                Ok(Subject::Map(subject))
            }
            State::Tuple(tuple) => {
                let subject = tuple
                    .into_iter()
                    .map(Subject::from_state)
                    .collect::<Result<Tuple<Subject>, E>>()?;

                Ok(Subject::Tuple(subject))
            }

            other => Err(de::Error::invalid_type(other.class(), ERR_INVALID)),
        }
    }

    /// Create a new `Subject` with the given `Schema`.
    pub fn create<'a>(schema: Schema, dir: &'a fs::Dir, txn_id: TxnId) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            match schema {
                Schema::Collection(schema) => {
                    SubjectCollection::create(schema, dir, txn_id)
                        .map_ok(Self::Collection)
                        .await
                }
                Schema::Tuple(schema) => {
                    try_join_all(
                        schema
                            .into_iter()
                            .enumerate()
                            .map(|(i, schema)| async move {
                                let dir = dir.create_dir(txn_id, i.into()).await?;
                                Self::create(schema, &dir, txn_id).await
                            }),
                    )
                    .map_ok(Tuple::from)
                    .map_ok(Self::Tuple)
                    .await
                }
                Schema::Map(schema) => {
                    try_join_all(schema.into_iter().map(|(name, schema)| async move {
                        let dir = dir.create_dir(txn_id, name.clone()).await?;
                        Self::create(schema, &dir, txn_id)
                            .map_ok(|subject| (name, subject))
                            .await
                    }))
                    .map_ok(|schemata| schemata.into_iter().collect())
                    .map_ok(Self::Map)
                    .await
                }
            }
        })
    }

    pub(super) fn load<'a>(
        txn: &'a Txn,
        schema: Schema,
        dir: &'a fs::Dir,
    ) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            let txn_id = *txn.id();

            match schema {
                Schema::Collection(schema) => {
                    SubjectCollection::load(txn, schema, dir)
                        .map_ok(Self::Collection)
                        .await
                }

                Schema::Map(schema) if schema.is_empty() => {
                    if let Some(file) = dir
                        .get_file::<fs::File<Scalar>, Scalar>(txn_id, &DYNAMIC.into())
                        .await?
                    {
                        let mut map = Map::new();
                        for id in file.block_ids(txn_id).await? {
                            let schema = file.read_block(txn_id, id.clone()).await?;
                            let schema = Schema::from_scalar(Scalar::clone(&*schema))?;
                            let subject = Self::load(txn, schema, dir).await?;
                            map.insert(id, subject);
                        }

                        Ok(Self::Map(map))
                    } else {
                        let file: fs::File<Scalar> = dir
                            .create_file(txn_id, DYNAMIC.into(), ScalarType::default())
                            .await?;

                        let schema = Scalar::Map(Map::default());
                        file.create_block(txn_id, DYNAMIC.into(), schema, 2).await?;

                        Ok(Self::Map(Map::default()))
                    }
                }
                Schema::Map(schema) => {
                    try_join_all(schema.into_iter().map(|(name, schema)| async move {
                        if let Some(dir) = dir.get_dir(txn_id, &name).await? {
                            Self::load(txn, schema, &dir)
                                .map_ok(|subject| (name, subject))
                                .await
                        } else {
                            let dir = dir.create_dir(txn_id, name.clone()).await?;
                            Self::create(schema, &dir, txn_id)
                                .map_ok(|subject| (name, subject))
                                .await
                        }
                    }))
                    .map_ok(|subjects| subjects.into_iter().collect())
                    .map_ok(Self::Map)
                    .await
                }

                Schema::Tuple(schema) => {
                    try_join_all(
                        schema
                            .into_iter()
                            .enumerate()
                            .map(|(i, schema)| async move {
                                if let Some(dir) = dir.get_dir(txn_id, &i.into()).await? {
                                    Self::load(txn, schema, &dir).await
                                } else {
                                    let dir = dir.create_dir(txn_id, i.into()).await?;
                                    Self::create(schema, &dir, txn_id).await
                                }
                            }),
                    )
                    .map_ok(Tuple::from)
                    .map_ok(Self::Tuple)
                    .await
                }
            }
        })
    }

    pub(super) fn restore<'a>(&'a self, txn: &'a Txn, backup: State) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            match self {
                Self::Collection(subject) => subject.restore(txn, backup).await,

                Self::Map(map) => match backup {
                    State::Map(mut backups) => {
                        let backups = map
                            .iter()
                            .map(|(name, subject)| {
                                backups
                                    .remove(name)
                                    .ok_or_else(|| {
                                        TCError::bad_request(
                                            "backup not found for Chain subject",
                                            name,
                                        )
                                    })
                                    .map(|backup| (subject, backup))
                            })
                            .collect::<TCResult<Vec<(&Subject, State)>>>()?;

                        let restores = backups
                            .into_iter()
                            .map(|(subject, backup)| subject.restore(txn, backup));

                        try_join_all(restores).await?;
                        Ok(())
                    }
                    backup => Err(TCError::unsupported(format!(
                        "invalid backup for schema {}: {}",
                        map, backup
                    ))),
                },

                Self::Tuple(tuple) => match backup {
                    State::Tuple(backup) if backup.len() == tuple.len() => {
                        let restores =
                            tuple
                                .iter()
                                .zip(backup)
                                .map(|(subject, backup)| async move {
                                    subject.restore(txn, backup).await
                                });

                        try_join_all(restores).await?;
                        Ok(())
                    }
                    State::Tuple(_) => Err(TCError::bad_request(
                        "backup has the wrong number of subjects for schema",
                        tuple,
                    )),
                    backup => Err(TCError::unsupported(format!(
                        "invalid backup for schema {}: {}",
                        tuple, backup
                    ))),
                },
            }
        })
    }

    pub fn hash<'a>(self, txn: Txn) -> TCBoxTryFuture<'a, Output<Sha256>> {
        Box::pin(async move {
            // TODO: should this be consolidated with Collection::hash?
            match self {
                Self::Collection(subject) => subject.hash(txn).await,

                Self::Map(map) => {
                    let mut hasher = Sha256::default();
                    for (id, subject) in map {
                        let subject = subject.hash(txn.clone()).await?;

                        let mut inner_hasher = Sha256::default();
                        inner_hasher.update(&Hash::<Sha256>::hash(id));
                        inner_hasher.update(&subject);
                        hasher.update(&inner_hasher.finalize());
                    }

                    Ok(hasher.finalize())
                }

                Self::Tuple(tuple) => {
                    let mut hasher = Sha256::default();
                    for subject in tuple {
                        let subject = subject.hash(txn.clone()).await?;
                        hasher.update(&subject);
                    }
                    Ok(hasher.finalize())
                }
            }
        })
    }
}

impl Instance for Subject {
    type Class = StateType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Collection(subject) => subject.class(),
            Self::Map(_) => StateType::Map,
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

#[async_trait]
impl Transact for Subject {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain subject");

        match self {
            Self::Collection(subject) => subject.commit(txn_id).await,
            Self::Map(map) => {
                join_all(
                    map.iter()
                        .map(|(_, subject)| async move { subject.commit(txn_id).await }),
                )
                .await;
            }
            Self::Tuple(tuple) => {
                join_all(
                    tuple
                        .iter()
                        .map(|subject| async move { subject.commit(txn_id).await }),
                )
                .await;
            }
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize chain subject");

        match self {
            Self::Collection(subject) => subject.finalize(txn_id).await,
            Self::Map(map) => {
                join_all(map.iter().map(|(_, subject)| async move {
                    subject.finalize(txn_id).await;
                }))
                .await;
            }
            Self::Tuple(tuple) => {
                join_all(tuple.iter().map(|subject| async move {
                    subject.finalize(txn_id).await;
                }))
                .await;
            }
        }
    }
}

#[async_trait]
impl de::FromStream for Subject {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let state = State::from_stream(txn, decoder).await?;
        Self::from_state(state)
    }
}

impl From<SubjectCollection> for Subject {
    fn from(subject: SubjectCollection) -> Self {
        Self::Collection(subject)
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Subject {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Collection(subject) => subject.into_view(txn).await,
            Self::Map(map) => {
                let views = map.into_iter().map(|(name, subject)| {
                    let txn = txn.clone();
                    async move { subject.into_view(txn).map_ok(|view| (name, view)).await }
                });

                let views = try_join_all(views).await?;
                Ok(StateView::Map(views.into_iter().collect()))
            }
            Self::Tuple(tuple) => {
                let views = tuple
                    .into_iter()
                    .map(|subject| subject.into_view(txn.clone()));

                try_join_all(views).map_ok(StateView::Tuple).await
            }
        }
    }
}

impl From<Subject> for State {
    fn from(subject: Subject) -> Self {
        match subject {
            Subject::Collection(subject) => State::Collection(subject.into()),
            Subject::Map(map) => State::Map(
                map.into_iter()
                    .map(|(name, subject)| (name, State::from(subject)))
                    .collect(),
            ),
            Subject::Tuple(tuple) => State::Tuple(tuple.into_iter().map(State::from).collect()),
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Collection(subject) => fmt::Display::fmt(subject, f),
            Self::Map(map) => fmt::Display::fmt(map, f),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
        }
    }
}
