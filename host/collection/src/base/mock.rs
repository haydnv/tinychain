use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use freqfs::FileSave;
use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::hash::{default_hash, AsyncHash, Output, Sha256};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::{Instance, NativeClass, TCPathBuf, ThreadSafe};

use crate::{Collection, CollectionType, CollectionView, Schema};

/// The base type of a mutable transactional collection of data.
pub struct CollectionBase<Txn, FE> {
    pub(crate) phantom: PhantomData<(Txn, FE)>,
}

impl<Txn, FE> Clone for CollectionBase<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            phantom: self.phantom,
        }
    }
}

impl<Txn, FE> CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: ThreadSafe,
{
    /// Return the [`Schema`] of this [`Collection`]
    pub fn schema(&self) -> Schema {
        Schema
    }
}

impl<Txn, FE> Instance for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: ThreadSafe,
{
    type Class = CollectionType;

    fn class(&self) -> CollectionType {
        CollectionType
    }
}

#[async_trait]
impl<Txn, FE> Transact for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'en> FileSave<'en> + Clone,
{
    type Commit = ();

    async fn commit(&self, _txn_id: TxnId) -> Self::Commit {}

    async fn rollback(&self, _txn_id: &TxnId) {}

    async fn finalize(&self, _txn_id: &TxnId) {}
}

#[async_trait]
impl<T, FE> AsyncHash for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: ThreadSafe + Clone,
{
    async fn hash(&self, _txn_id: TxnId) -> TCResult<Output<Sha256>> {
        Ok(default_hash::<Sha256>())
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + Clone,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, _schema: Schema, _store: Dir<FE>) -> TCResult<Self> {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    async fn load(_txn_id: TxnId, _schema: Schema, _store: Dir<FE>) -> TCResult<Self> {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    fn dir(&self) -> tc_transact::fs::Inner<FE> {
        unimplemented!("mock collection dir")
    }
}

#[async_trait]
impl<Txn, FE> CopyFrom<FE, Collection<Txn, FE>> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + Clone,
{
    async fn copy_from(
        _txn: &Txn,
        _store: Dir<FE>,
        _instance: Collection<Txn, FE>,
    ) -> TCResult<Self> {
        Ok(Self {
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<Txn, FE> Restore<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + Clone,
{
    async fn restore(&self, _txn_id: TxnId, _backup: &Self) -> TCResult<()> {
        Ok(())
    }
}

impl<T, FE> TryCastFrom<Collection<T, FE>> for CollectionBase<T, FE> {
    fn can_cast_from(_collection: &Collection<T, FE>) -> bool {
        true
    }

    fn opt_cast_from(_collection: Collection<T, FE>) -> Option<Self> {
        Some(Self {
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: ThreadSafe + Clone,
    Self: 'en,
{
    type Txn = T;
    type View = CollectionView<'en>;

    async fn into_view(self, _txn: Self::Txn) -> TCResult<Self::View> {
        Ok(CollectionView::default())
    }
}

impl<T, FE> fmt::Debug for CollectionBase<T, FE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Collection")
    }
}

/// A [`de::Visitor`] used to deserialize a [`Collection`].
pub struct CollectionVisitor<Txn, FE> {
    #[allow(unused)]
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> CollectionVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: for<'a> FileSave<'a> + Clone,
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
        _access: &mut A,
    ) -> Result<CollectionBase<Txn, FE>, A::Error> {
        debug!("CollectionVisitor::visit_map_value with class {:?}", class);
        Ok(CollectionBase {
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<T, FE> de::Visitor for CollectionVisitor<T, FE>
where
    T: Transaction<FE>,
    FE: for<'a> FileSave<'a> + Clone,
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
    FE: for<'a> FileSave<'a> + Clone,
{
    type Context = T;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor::new(txn)).await
    }
}
