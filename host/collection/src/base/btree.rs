use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use freqfs::FileSave;
use futures::TryFutureExt;
use log::{debug, info};
use safecast::{AsType, TryCastFrom};

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::hash::{AsyncHash, Output, Sha256};
use tc_transact::{fs, IntoView, Transact, Transaction, TxnId};
use tcgeneric::{Instance, NativeClass, TCPathBuf, ThreadSafe};

use crate::btree::{BTree, BTreeFile, BTreeInstance, Node as BTreeNode};
use crate::{Collection, CollectionType, CollectionView, Schema};

/// The base type of a mutable transactional collection of data.
pub enum CollectionBase<Txn, FE> {
    BTree(BTreeFile<Txn, FE>),
}

impl<Txn, FE> Clone for CollectionBase<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::BTree(btree) => Self::BTree(btree.clone()),
        }
    }
}

impl<Txn, FE> CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<BTreeNode> + ThreadSafe,
{
    /// Return the [`Schema`] of this [`Collection`]
    pub fn schema(&self) -> Schema {
        match self {
            Self::BTree(btree) => btree.schema().clone().into(),
        }
    }
}

impl<Txn, FE> Instance for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: ThreadSafe,
{
    type Class = CollectionType;

    fn class(&self) -> CollectionType {
        match self {
            Self::BTree(btree) => btree.class().into(),
        }
    }
}

#[async_trait]
impl<Txn, FE> Transact for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<BTreeNode> + for<'en> fs::FileSave<'en> + Clone,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl<T, FE> AsyncHash for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: AsType<BTreeNode> + ThreadSafe + Clone,
{
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        Collection::from(self.clone()).hash(txn_id).await
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + AsType<BTreeNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        debug!("create persistent mutable collection at {store:?}");

        match schema {
            Schema::BTree(schema) => {
                BTreeFile::create(txn_id, schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
        }
    }

    async fn load(txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        info!("load persistent mutable collection at {store:?}");

        match schema {
            Schema::BTree(schema) => {
                BTreeFile::load(txn_id, schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
        }
    }

    fn dir(&self) -> tc_transact::fs::Inner<FE> {
        match self {
            Self::BTree(btree) => btree.dir(),
        }
    }
}

#[async_trait]
impl<Txn, FE> CopyFrom<FE, Collection<Txn, FE>> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + AsType<BTreeNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    async fn copy_from(txn: &Txn, store: Dir<FE>, instance: Collection<Txn, FE>) -> TCResult<Self> {
        match instance {
            Collection::BTree(instance) => {
                BTreeFile::copy_from(txn, store, instance)
                    .map_ok(Self::BTree)
                    .await
            }
        }
    }
}

#[async_trait]
impl<Txn, FE> Restore<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + AsType<BTreeNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        match (self, backup) {
            (Self::BTree(this), Self::BTree(backup)) => this.restore(txn_id, backup).await,
        }
    }
}

impl<T, FE> TryCastFrom<Collection<T, FE>> for CollectionBase<T, FE> {
    fn can_cast_from(collection: &Collection<T, FE>) -> bool {
        match collection {
            Collection::BTree(BTree::File(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<T, FE>) -> Option<Self> {
        match collection {
            Collection::BTree(BTree::File(btree)) => Some(Self::BTree(btree)),
            _ => None,
        }
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: AsType<BTreeNode> + ThreadSafe + Clone,
    Self: 'en,
{
    type Txn = T;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        Collection::from(self).into_view(txn).await
    }
}

impl<T, FE> fmt::Debug for CollectionBase<T, FE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Collection")
    }
}

/// A [`de::Visitor`] used to deserialize a [`Collection`].
pub struct CollectionVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> CollectionVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + AsType<BTreeNode> + Clone,
{
    pub fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }

    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: CollectionType,
        access: &mut A,
    ) -> Result<CollectionBase<Txn, FE>, A::Error> {
        debug!("CollectionVisitor::visit_map_value with class {:?}", class);

        match class {
            CollectionType::BTree(_) => {
                access
                    .next_value(self.txn)
                    .map_ok(CollectionBase::BTree)
                    .await
            }
        }
    }
}

#[async_trait]
impl<T, FE> de::Visitor for CollectionVisitor<T, FE>
where
    T: Transaction<FE>,
    FE: for<'a> FileSave<'a> + AsType<BTreeNode> + Clone,
{
    type Value = CollectionBase<T, FE>;

    fn expecting() -> &'static str {
        "a Collection"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let classpath: TCPathBuf = map
            .next_key(())
            .await?
            .ok_or_else(|| de::Error::custom("expected a Collection type"))?;

        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| de::Error::invalid_value(classpath, "a Collection type"))?;

        self.visit_map_value(class, &mut map).await
    }
}

#[async_trait]
impl<T, FE> de::FromStream for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: for<'a> FileSave<'a> + AsType<BTreeNode> + Clone,
{
    type Context = T;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor::new(txn)).await
    }
}
