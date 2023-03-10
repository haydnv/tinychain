use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::{IntoView, Transaction, TxnId};
use tcgeneric::{NativeClass, TCPathBuf, ThreadSafe};

use super::tensor::TensorType;
use super::{Collection, CollectionType, CollectionView, Schema};

#[derive(Clone)]
pub struct CollectionBase<T, FE> {
    phantom: PhantomData<(T, FE)>,
}

#[async_trait]
impl<T: Transaction<FE>, FE: ThreadSafe> Persist<FE> for CollectionBase<T, FE> {
    type Txn = T;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, _schema: Schema, _store: Dir<FE>) -> TCResult<Self> {
        todo!()
    }

    async fn load(_txn_id: TxnId, _schema: Schema, _store: Dir<FE>) -> TCResult<Self> {
        todo!()
    }

    fn dir(&self) -> &tc_transact::fs::Inner<FE> {
        todo!()
    }
}

#[async_trait]
impl<T, FE> CopyFrom<FE, Collection<T, FE>> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: ThreadSafe,
{
    async fn copy_from(_txn: &T, _store: Dir<FE>, _instance: Collection<T, FE>) -> TCResult<Self> {
        todo!()
    }
}

#[async_trait]
impl<T: Transaction<FE>, FE: ThreadSafe> Restore<FE> for CollectionBase<T, FE> {
    async fn restore(&self, _txn_id: TxnId, _backup: &Self) -> TCResult<()> {
        todo!()
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: Send + Sync,
    Self: 'en,
{
    type Txn = T;
    type View = CollectionView<'en, T, FE>;

    async fn into_view(self, _txn: Self::Txn) -> TCResult<Self::View> {
        todo!()
    }
}

impl<T, FE> fmt::Debug for CollectionBase<T, FE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Collection")
    }
}

/// A [`de::Visitor`] used to deserialize a [`Collection`].
pub struct CollectionVisitor<T, FE> {
    #[allow(unused)]
    txn: T,
    phantom: PhantomData<FE>,
}

impl<T, FE> CollectionVisitor<T, FE> {
    pub fn new(txn: T) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }

    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: CollectionType,
        _access: &mut A,
    ) -> Result<CollectionBase<T, FE>, A::Error> {
        match class {
            CollectionType::BTree(_) => {
                todo!()
            }

            CollectionType::Table(_) => {
                todo!()
            }

            CollectionType::Tensor(tt) => match tt {
                TensorType::Dense => {
                    todo!()
                }
                TensorType::Sparse => {
                    todo!()
                }
            },
        }
    }
}

#[async_trait]
impl<T, FE> de::Visitor for CollectionVisitor<T, FE>
where
    T: Transaction<FE>,
    FE: Send + Sync,
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
    FE: Send + Sync,
{
    type Context = T;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor::new(txn)).await
    }
}
