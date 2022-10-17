use std::convert::TryInto;

use async_trait::async_trait;
use futures::future::{FutureExt, TryFutureExt};

use tc_error::*;
use tc_transact::fs::{CopyFrom, Persist};
use tc_transact::Transact;
use tcgeneric::Instance;

use crate::fs;
use crate::transact::TxnId;
use crate::txn::Txn;

use super::schema::{CollectionSchema, CollectionType};
use super::{BTreeFile, Collection, TableIndex};
#[cfg(feature = "tensor")]
use super::{DenseTensorFile, SparseTable, TensorType};

/// The base type of a [`Collection`] which supports [`CopyFrom`], [`Persist`], and [`Transact`]
pub enum CollectionBase {
    BTree(BTreeFile),
    Table(TableIndex),
    #[cfg(feature = "tensor")]
    Dense(DenseTensorFile),
    #[cfg(feature = "tensor")]
    Sparse(SparseTable),
}

impl Instance for CollectionBase {
    type Class = CollectionType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => btree.class().into(),
            Self::Table(table) => table.class().into(),
            #[cfg(feature = "tensor")]
            Self::Dense(_dense) => TensorType::Dense.into(),
            #[cfg(feature = "tensor")]
            Self::Sparse(_sparse) => TensorType::Sparse.into(),
        }
    }
}

#[async_trait]
impl Persist<fs::Dir> for CollectionBase {
    type Schema = CollectionSchema;
    type Store = fs::Store;
    type Txn = Txn;

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        match schema {
            CollectionSchema::BTree(btree_schema) => {
                let store = store.try_into()?;
                BTreeFile::load(txn, btree_schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            CollectionSchema::Table(table_schema) => {
                let store = store.try_into()?;
                TableIndex::load(txn, table_schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(tensor_schema) => {
                let store = store.try_into()?;
                DenseTensorFile::load(txn, tensor_schema, store)
                    .map_ok(Self::Dense)
                    .await
            }
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(tensor_schema) => {
                let store = store.try_into()?;
                SparseTable::load(txn, tensor_schema, store)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }
}

#[async_trait]
impl CopyFrom<fs::Dir, Collection> for CollectionBase {
    async fn copy_from(instance: Collection, store: Self::Store, txn: &Txn) -> TCResult<Self> {
        match instance {
            Collection::BTree(btree) => {
                let store = store.try_into()?;
                BTreeFile::copy_from(btree, store, txn)
                    .map_ok(Self::BTree)
                    .await
            }
            Collection::Table(table) => {
                let store = store.try_into()?;
                TableIndex::copy_from(table, store, txn)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Collection::Dense(dense) => {
                let store = store.try_into()?;
                DenseTensorFile::copy_from(dense, store, txn)
                    .map_ok(Self::Dense)
                    .await
            }
            #[cfg(feature = "tensor")]
            Collection::Sparse(sparse) => {
                let store = store.try_into()?;
                SparseTable::copy_from(btree, store, txn)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }
}

pub enum CollectionBaseCommitGuard {
    BTree(<BTreeFile as Transact>::Commit),
    Table(<TableIndex as Transact>::Commit),
    #[cfg(feature = "tensor")]
    Dense(<DenseTensorFile as Transact>::Commit),
    #[cfg(feature = "tensor")]
    Sparse(<SparseTable as Transact>::Commit),
}

#[async_trait]
impl Transact for CollectionBase {
    type Commit = CollectionBaseCommitGuard;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => {
                btree
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::BTree)
                    .await
            }
            Self::Table(table) => {
                table
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => {
                dense
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::Dense)
                    .await
            }
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => {
                sparse
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::Sparse)
                    .await
            }
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => dense.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
        }
    }
}
