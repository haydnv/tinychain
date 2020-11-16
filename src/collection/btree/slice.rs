use async_trait::async_trait;
use futures::stream::StreamExt;

use crate::class::{Instance, State, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::{validate_range, BTree, BTreeFile, BTreeRange, BTreeType, Key, Selector};

#[derive(Clone)]
pub struct BTreeSlice {
    source: BTreeFile,
    bounds: Selector,
}

impl BTreeSlice {
    pub fn new(source: BTreeFile, bounds: Selector) -> BTreeSlice {
        BTreeSlice { source, bounds }
    }
}

impl Instance for BTreeSlice {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        BTreeType::Slice
    }
}

#[async_trait]
impl CollectionInstance for BTreeSlice {
    type Item = Key;
    type Slice = BTreeSlice;

    async fn get(
        &self,
        _request: &Request,
        _txn: &Txn,
        path: &[PathSegment],
        range: Value,
    ) -> TCResult<State> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let range: BTreeRange =
            range.try_cast_into(|v| error::bad_request("Invalid BTree range", v))?;
        let range = validate_range(range, self.source.schema())?;

        let schema: Vec<ValueType> = self.source.schema().iter().map(|c| *c.dtype()).collect();
        let selector = Selector::from(range); // TODO: support reverse order

        if self.bounds == selector {
            Ok(State::Collection(Collection::View(self.clone().into())))
        } else if self.bounds.range().contains(selector.range(), &schema)? {
            Err(error::not_implemented("BTreeSlice::get slice"))
        } else {
            let bounds: Value = self.bounds.range().clone().cast_into();
            let selector: Value = selector.range().clone().cast_into();
            Err(error::bad_request(
                format!("BTreeSlice[{}] does not contain", &bounds),
                &selector,
            ))
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let count = self
            .source
            .clone()
            .len(txn.id().clone(), self.bounds.clone())
            .await?;

        Ok(count == 0)
    }

    async fn put(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _selector: Value,
        _value: State,
    ) -> TCResult<()> {
        Err(error::unsupported(
            "BTreeSlice is immutable; write to the source BTree instead",
        ))
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        let rows = self
            .source
            .clone()
            .slice(txn.id().clone(), self.bounds.clone())
            .await?;

        Ok(Box::pin(rows.map(Value::Tuple).map(Scalar::Value)))
    }
}

#[async_trait]
impl Transact for BTreeSlice {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

impl From<BTree> for BTreeSlice {
    fn from(btree: BTree) -> BTreeSlice {
        match btree {
            BTree::View(slice) => slice,
            BTree::Tree(btree) => BTreeSlice {
                source: btree,
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
        CollectionView::BTree(BTree::View(btree_slice))
    }
}
