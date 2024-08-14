use std::marker::PhantomData;

use async_trait::async_trait;
use collate::OverlapsRange;
use futures::{future, TryFutureExt, TryStreamExt};
use safecast::AsType;

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tcgeneric::{Instance, ThreadSafe};

use super::file::BTreeFile;
use super::schema::BTreeSchema;
use super::stream::Keys;
use super::{BTreeInstance, BTreeType, Node, Range};

pub struct BTreeSlice<Txn, FE> {
    file: BTreeFile<Txn, FE>,
    range: Range,
    reverse: bool,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE> Clone for BTreeSlice<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            file: self.file.clone(),
            range: self.range.clone(),
            reverse: self.reverse,
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> BTreeSlice<Txn, FE> {
    pub(super) fn new(file: BTreeFile<Txn, FE>, range: Range, reverse: bool) -> Self {
        Self {
            file,
            range,
            reverse,
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> Instance for BTreeSlice<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
{
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        BTreeType::Slice
    }
}

#[async_trait]
impl<Txn, FE> BTreeInstance for BTreeSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = Self;

    fn schema(&self) -> &BTreeSchema {
        self.file.schema()
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        let range = self.schema().validate_range(range)?;

        if self.range.contains(&range, self.file.collator()) {
            Ok(Self::new(self.file, range, self.reverse ^ reverse))
        } else {
            Err(bad_request!(
                "slice {:?} does not contain the range {:?}",
                self.range,
                range
            ))
        }
    }

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let keys = self.clone().keys(txn_id).await?;
        keys.try_fold(0u64, |count, _key| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        let mut keys = self.clone().keys(txn_id).await?;
        keys.try_next().map_ok(|key| key.is_none()).await
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<Keys<'a>>
    where
        Self: 'a,
    {
        self.file
            .into_stream(txn_id, self.range, self.reverse)
            .await
    }
}
