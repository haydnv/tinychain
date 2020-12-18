use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};

use log::debug;

use crate::class::Instance;
use crate::collection::schema::Column;
use crate::collection::Collection;
use crate::error;
use crate::general::{TCResult, TCStream};
use crate::transaction::{Transact, Txn, TxnId};

use super::{BTree, BTreeFile, BTreeImpl, BTreeInstance, BTreeRange, BTreeType, Key};

pub const ERR_BOUNDS: &str = "Requested range is outside the bounds of the containing view \
(hint: try slicing the base BTree instead)";

const ERR_INSERT: &str = "BTreeSlice does not support insert";

#[derive(Clone)]
pub struct BTreeSlice {
    source: BTreeImpl<BTreeFile>,
    range: BTreeRange,
    reverse: bool,
}

impl BTreeSlice {
    pub fn new(source: BTree, range: BTreeRange, reverse: bool) -> TCResult<BTreeSlice> {
        match source {
            BTree::Tree(tree) => {
                debug!(
                    "BTreeSlice from source tree with range {} (reverse: {})",
                    range, reverse
                );
                Ok(Self {
                    source: tree,
                    range,
                    reverse,
                })
            }
            BTree::View(view) => {
                let source = view.source.clone();
                let reverse = view.reverse ^ reverse;
                debug!(
                    "BTreeSlice from view with range {} (reverse: {})",
                    range, reverse
                );

                if range == BTreeRange::default() {
                    Ok(Self {
                        source,
                        range: view.into_inner().range,
                        reverse,
                    })
                } else if view
                    .range
                    .contains(&range, view.schema(), source.collator())
                {
                    Ok(Self {
                        source,
                        range,
                        reverse,
                    })
                } else {
                    Err(error::bad_request(ERR_BOUNDS, range))
                }
            }
        }
    }
}

impl Instance for BTreeSlice {
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        BTreeType::View
    }
}

#[async_trait]
impl BTreeInstance for BTreeSlice {
    async fn delete(&self, txn_id: &TxnId, range: BTreeRange) -> TCResult<()> {
        if range == BTreeRange::default() {
            self.source.delete(txn_id, self.range.clone()).await
        } else if self
            .range
            .contains(&range, self.schema(), self.source.collator())
        {
            self.source.delete(txn_id, range).await
        } else {
            Err(error::bad_request(ERR_BOUNDS, range))
        }
    }

    async fn insert(&self, _txn_id: &TxnId, _key: Key) -> TCResult<()> {
        Err(error::unsupported(ERR_INSERT))
    }

    async fn insert_from<S: Stream<Item = Key> + Send>(
        &self,
        _txn_id: &TxnId,
        _source: S,
    ) -> TCResult<()> {
        Err(error::unsupported(ERR_INSERT))
    }

    async fn try_insert_from<S: Stream<Item = TCResult<Key>> + Send>(
        &self,
        _txn_id: &TxnId,
        _source: S,
    ) -> TCResult<()> {
        Err(error::unsupported(ERR_INSERT))
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let mut rows = self
            .stream(txn.id(), self.range.clone(), self.reverse)
            .await?;

        let empty = rows.next().await.is_none();
        Ok(empty)
    }

    async fn len(&self, txn_id: TxnId, range: BTreeRange) -> TCResult<u64> {
        if range == BTreeRange::default() {
            self.source.len(txn_id, self.range.clone()).await
        } else if self
            .range
            .contains(&range, self.schema(), self.source.collator())
        {
            self.source.len(txn_id, range).await
        } else {
            Err(error::bad_request(ERR_BOUNDS, range))
        }
    }

    fn schema(&'_ self) -> &'_ [Column] {
        self.source.schema()
    }

    async fn stream<'a>(
        &'a self,
        txn_id: &'a TxnId,
        range: BTreeRange,
        reverse: bool,
    ) -> TCResult<TCStream<'a, Key>> {
        debug!(
            "reverse: {} ^ {} = {}",
            reverse,
            self.reverse,
            reverse ^ self.reverse
        );
        let reverse = reverse ^ self.reverse;

        if range == BTreeRange::default() {
            debug!("BTreeSlice::slice {} (reverse: {})", &self.range, reverse);
            self.source
                .stream(txn_id, self.range.clone(), reverse)
                .await
        } else if self
            .range
            .contains(&range, self.schema(), self.source.collator())
        {
            debug!(
                "BTreeSlice::slice with constrained bounds: {} (reverse: {})",
                range, reverse
            );
            self.source.stream(txn_id, range, reverse).await
        } else {
            Err(error::bad_request(ERR_BOUNDS, range))
        }
    }
}

#[async_trait]
impl Transact for BTreeSlice {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

impl From<BTreeSlice> for Collection {
    fn from(slice: BTreeSlice) -> Collection {
        Collection::BTree(slice.into())
    }
}
