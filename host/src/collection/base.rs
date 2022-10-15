use async_trait::async_trait;
use futures::future::FutureExt;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Persist};
use tc_transact::Transact;
use tcgeneric::Instance;

use crate::fs;
use crate::fs::DirEntry;
use crate::transact::TxnId;
use crate::txn::Txn;

use super::schema::{CollectionSchema, CollectionType};
use super::{BTreeFile, Collection, TableIndex};
#[cfg(feature = "tensor")]
use super::{DenseTensorFile, SparseTable};

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
            Self::Dense(dense) => dense.class().into(),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.class().into(),
        }
    }
}

#[async_trait]
impl Persist<fs::Dir> for CollectionBase {
    type Schema = CollectionSchema;
    type Store = fs::DirEntry;
    type Txn = Txn;

    fn schema(&self) -> &Self::Schema {
        todo!()
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        todo!()
    }
}

#[async_trait]
impl CopyFrom<fs::Dir, Collection> for CollectionBase {
    async fn copy_from(instance: Collection, store: DirEntry, txn: &Txn) -> TCResult<Self> {
        todo!()
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
