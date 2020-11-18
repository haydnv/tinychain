use async_trait::async_trait;
use futures::stream::Stream;

use log::debug;

use crate::class::{TCResult, TCStream};
use crate::collection::schema::Column;
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};

use super::{BTree, BTreeFile, BTreeInstance, BTreeRange, Key, Selector};

pub const ERR_BOUNDS: &str = "Requested range is outside the bounds of the containing view \
(hint: try slicing the base BTree instead)";

const ERR_INSERT: &str = "BTreeSlice does not support insert";

#[derive(Clone)]
pub struct BTreeSlice {
    source: BTreeFile,
    bounds: Selector,
}

impl BTreeSlice {
    pub fn new(source: BTreeFile, bounds: Selector) -> BTreeSlice {
        assert!(bounds != Selector::default());

        debug!("new BTreeSlice with bounds {}", bounds);

        BTreeSlice { source, bounds }
    }

    pub fn selector(&'_ self) -> &'_ Selector {
        &self.bounds
    }

    pub fn source(&'_ self) -> &'_ BTreeFile {
        &self.source
    }
}

#[async_trait]
impl BTreeInstance for BTreeSlice {
    async fn delete(&self, txn_id: &TxnId, range: BTreeRange) -> TCResult<()> {
        if range == BTreeRange::default() {
            self.source
                .delete(txn_id, self.bounds.range().clone())
                .await
        } else if self.bounds.range().contains(&range, self.schema())? {
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
        let count = self
            .source
            .clone()
            .len(txn.id().clone(), self.bounds.range().clone())
            .await?;

        Ok(count == 0)
    }

    async fn len(&self, txn_id: TxnId, range: BTreeRange) -> TCResult<u64> {
        if range == BTreeRange::default() {
            self.source.len(txn_id, self.bounds.range().clone()).await
        } else if self.bounds.range().contains(&range, self.schema())? {
            self.source.len(txn_id, range).await
        } else {
            Err(error::bad_request(ERR_BOUNDS, range))
        }
    }

    fn schema(&'_ self) -> &'_ [Column] {
        self.source.schema()
    }

    async fn slice(&self, txn_id: TxnId, selector: Selector) -> TCResult<TCStream<Key>> {
        if selector.range() == &BTreeRange::default() {
            let reverse = selector.reverse() ^ self.bounds.reverse();

            debug!(
                "BTreeSlice::slice {} (reverse: {})",
                self.bounds.range(),
                reverse
            );

            self.source
                .slice(txn_id, (self.bounds.range().clone(), reverse).into())
                .await
        } else if self
            .bounds
            .range()
            .contains(selector.range(), self.schema())?
        {
            debug!("BTreeSlice::slice with constrained bounds");

            self.source.slice(txn_id, selector).await
        } else {
            Err(error::bad_request(ERR_BOUNDS, selector.range()))
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

impl From<BTree> for BTreeSlice {
    fn from(btree: BTree) -> BTreeSlice {
        match btree {
            BTree::View(slice) => slice.into_inner(),
            BTree::Tree(btree) => BTreeSlice {
                source: btree.into_inner(),
                bounds: Selector::default(),
            },
        }
    }
}

impl From<BTreeSlice> for Collection {
    fn from(btree_slice: BTreeSlice) -> Collection {
        Collection::View(btree_slice.into())
    }
}

impl From<BTreeSlice> for CollectionView {
    fn from(btree_slice: BTreeSlice) -> CollectionView {
        CollectionView::BTree(BTree::View(btree_slice.into()))
    }
}
